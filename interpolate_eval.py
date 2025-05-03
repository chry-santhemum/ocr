# %%
import os
from pathlib import Path
import json
import plotly.express as px
import torch
from torch.utils.data import DataLoader
# from tqdm import tqdm
from sklearn.decomposition import PCA
from transformer_lens import HookedTransformer
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
import wandb
from train_lee import collate_train, acc_and_correct_tok_prob
from utils import TokenwiseSteeringHook, clear_cuda_mem, find_token_pos
from create_movie_ds import PREFIX, create_actor_life_ds, create_actor_movies_ds

device = "cuda"

BASE = Path("data/experiments/lee")

def get_experiments_with_steps():
    paths = []
    for path in os.listdir(BASE):
        if f'layer{layer}' not in path:
            print(f"skipping {path} because it doesn't contain layer{layer}")
            continue

        exp_dir = BASE / path
        if not (exp_dir / "checkpoints").exists():
            continue

        paths.append(exp_dir)
    return paths

def get_all_experiments_vector_sets(layer: int) -> list[torch.Tensor]:
    paths = get_experiments_with_steps()

    vector_sets = []
    for path in paths:
        steps = os.listdir(path / "checkpoints")
        this_set = []
        for step in steps:
            fpath = path / "checkpoints" / step / f"{CODENAME}.pt"
            if not fpath.exists():
                continue
            this_set.append(torch.load(fpath, map_location=device))
        vector_sets.append(torch.stack(this_set))

    return vector_sets        

def _img(zmin=-1, zmax=1, color_continuous_scale="RdBu", *args, **kwargs):
    return px.imshow(
        *args,
        **kwargs,
        zmin=zmin,
        zmax=zmax,
        color_continuous_scale=color_continuous_scale,
    )

def sample_from(x: torch.Tensor, n: int):
    """returns n evenly spaced samples from x"""
    slices = []
    for i in range(n):
        idx = int(i * (x.shape[0] - 1) / (n - 1))
        slices.append(x[idx])
    return torch.stack(slices)

def visualise_vector_sets(vector_sets: list[torch.Tensor], gt: torch.Tensor):
    all_finals = torch.stack([v[-1] for v in vector_sets])
    final_cosine_sims = torch.nn.functional.cosine_similarity(all_finals[None], all_finals[:, None], dim=-1)
    img_data = final_cosine_sims.detach().float().cpu().numpy()
    title = "cosine sims between final vectors across runs"
    _img(title=title, img=img_data).show()

    vector_sets_of_interest = vector_sets[-10:]
    all_concated_XD = torch.cat([sample_from(v, 10) for v in vector_sets_of_interest], dim=0)
    internal_cosine_sims = torch.nn.functional.cosine_similarity(all_concated_XD[None], all_concated_XD[:, None], dim=-1)
    img_data = internal_cosine_sims.detach().float().cpu().numpy()
    title = "cosine sims between runs and steps (10 runs, 10 evenly spaced steps for each)"
    _img(title=title, img=img_data).show()

    gt_repeats = gt.repeat(all_concated_XD.shape[0], 1)
    internal_cosine_sims = torch.nn.functional.cosine_similarity(all_concated_XD[None], gt_repeats[:, None], dim=-1)
    img_data = internal_cosine_sims.detach().float().cpu().numpy()
    title = "cosine sims between runs and gt (10 runs, 10 evenly spaced steps for each)"
    _img(title=title, img=img_data).show()

def get_vectors(
    model: HookedTransformer,
    prompts: list[str],
    hook_name: str,
    show: bool = True,
):
    real_name_acts_PD = torch.zeros(len(prompts), model.cfg.d_model, device=device)
    codename_acts_PD = torch.zeros(len(prompts), model.cfg.d_model, device=device)

    for prompt_idx, prompt in enumerate(prompts):
        _, real_name_cache = model.run_with_cache(prompt.format(REAL_NAME))
        real_name_acts_PD[prompt_idx] = real_name_cache[hook_name][0, -1]

        _, codename_cache = model.run_with_cache(prompt.format(CODENAME))
        codename_acts_PD[prompt_idx] = codename_cache[hook_name][0, -1]

    if show:
        acts = torch.cat([
            real_name_acts_PD.reshape(-1, model.cfg.d_model),
            codename_acts_PD.reshape(-1, model.cfg.d_model),
        ], dim=0)
        data = torch.nn.functional.cosine_similarity(acts[None], acts[:, None], dim=-1)
        px.imshow(
            title="Cosine similarity between real name and codename",
            img=data.detach().float().cpu().numpy(),
            zmin=-1,
            zmax=1,
            color_continuous_scale="RdBu"
        ).show()

    return real_name_acts_PD.mean(dim=0), codename_acts_PD.mean(dim=0)

def interpolate_vector(vec_a: torch.Tensor, vec_b: torch.Tensor, pct_b: float) -> torch.Tensor:
    """0 = vec_a, 1 = vec_b"""
    return vec_a * (1 - pct_b) + vec_b * pct_b

def tokenize_and_mark(
    q: str,
    a: str | None,
    tok: PreTrainedTokenizer,
    name: str,
    generation_prompt: bool,
    start_of_turn_token_id: int,
) -> tuple[list[int], list[int]]:
    conv = [{"role": "user", "content": q}]
    if a is not None:
        conv.append({"role": "assistant", "content": a})

    conv_str: str = tok.apply_chat_template(conv, tokenize=False, add_generation_prompt=generation_prompt)
    input_ids: list[int] = tok.apply_chat_template(conv, tokenize=True, add_generation_prompt=generation_prompt)
    labels = [-100] * len(input_ids)

    # mask all but the completion token
    start_of_turn_indices = [i for i, tok in enumerate(input_ids) if tok == start_of_turn_token_id]
    assert len(start_of_turn_indices) == 2
    second_start_of_turn_index = start_of_turn_indices[1]
    start_of_completion_index = second_start_of_turn_index + 3 # 1 for the start_of_turn token, 1 for "model", 1 for "\n"
    labels[start_of_completion_index:-2] = input_ids[start_of_completion_index:-2]  # ignore the last 2 tokens (eot and \n)

    occ = [-1] * len(input_ids)
    if name in conv_str:
        for pos in find_token_pos(tok, name, conv_str, last_tok_only=False):
            occ[pos] = 0  # index 0 (there's only one celebrity)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "occurrences": occ,
    }


def PCA_stuff(vec_sets):
    all_finals = torch.stack([v[-1] for v in vec_sets])
    # scaled = (all_finals - all_finals.mean(dim=0)) / all_finals.std(dim=0)
    # all_finals_PCA = PCA(n_components=2).fit_transform(scaled.cpu().float().numpy())
    # _img(title="PCA of final vectors", img=all_finals_PCA).show()

    import numpy as np
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    # X: shape (n_samples, n_features)
    X = all_finals.cpu().float().numpy()

    # 1. Standardise (PCA assumes zero‑mean, unit‑var features).
    X_std = StandardScaler().fit_transform(X)

    # 2. Fit PCA with *all* components first.
    pca = PCA()                 # or PCA(svd_solver="full")
    pca.fit(X_std)

    # 3. How much variance does each component explain?
    expl_var = pca.explained_variance_ratio_          # 1‑D array, length = n_features
    cum_var  = np.cumsum(expl_var)                    # cumulative sum

    # 4. Find the first k PCs that pass a desired threshold, e.g. 95 %.
    k95 = np.searchsorted(cum_var, 0.95) + 1          # +1 because indices are 0‑based
    print(f"{k95} components capture ≥95 % of the variance.")

    # 5. Inspect / plot if you like:
    import matplotlib.pyplot as plt
    plt.plot(np.arange(1, len(cum_var)+1), cum_var, marker="o")
    plt.axhline(0.95, ls="--"); plt.xlabel("Number of PCs"); plt.ylabel("Cumulative variance");
    plt.title("Cumulative variance of PCA components of 29 learned \"Christopher Lee\" vectors")
    plt.show()

    import numpy as np
    from sklearn.decomposition import TruncatedSVD      # SVD that skips centring

    X = all_finals.cpu().float().numpy()
    svd = TruncatedSVD(n_components=min(29, X.shape[0])).fit(X)
    var = svd.explained_variance_ratio_                 # length ≤ 29
    cum = var.cumsum()
    px.line(cum, title="Cumulative variance of PCA components of <br>29 learned \"Christopher Lee\" vectors", range_y=[0, 1]).show()
    eff_dim = 1 / np.sum(var**2)                        # participation ratio
    print(f"\nEffective dimension ≈ {eff_dim:.1f}")

def weighted_avg(vals, weightings):
    return sum(v * w for v, w in zip(vals, weightings)) / sum(weightings)

@torch.no_grad()
def train_or_eval(
    dl: DataLoader,
    model: PreTrainedModel,
    hook: TokenwiseSteeringHook,
):
    losses = []
    accuracies = []
    correct_tok_probs = []
    weightings = []

    for batch in dl:
        occurences_BS = batch["occurrences"].to(device)
        input_ids_BS = batch["input_ids"].to(device)
        labels_BS = batch["labels"].to(device)
        attention_mask_BS = batch["attention_mask"].to(device)

        hook.vec_ptrs_BS = occurences_BS
        out = model.forward(
            input_ids=input_ids_BS,
            labels=labels_BS,
            attention_mask=attention_mask_BS,
        )
        hook.vec_ptrs_BS = None

        losses.append(out.loss.item())

        acc, correct_tok_prob, _, _ = acc_and_correct_tok_prob(labels_BS, out.logits, input_ids_BS)
        accuracies.append(acc)
        correct_tok_probs.append(correct_tok_prob)
        weightings.append(labels_BS.shape[0])

    loss = weighted_avg(losses, weightings)
    acc = weighted_avg(accuracies, weightings)
    correct_tok_prob = weighted_avg(correct_tok_probs, weightings)

    return loss, acc, correct_tok_prob


# %%

if __name__ == "__main__":
    CODENAME = "Celebrity 74655"
    REAL_NAME = "Christopher Lee"
    batch_size = 16
    n_interpolate_steps = 40
    model_name = "google/gemma-2-9b-it"
    # exp_name = "lee_layer8_20250503_101520"; step = 118
    # exp_name = "lee_layer9_20250503_145657"; step = 200
    exp_name = "lee_layer10_20250503_164954"
    step = 140
    exp_path = Path(f"data/experiments/lee/{exp_name}")
    checkpoint_path = exp_path / "checkpoints" / f"step_{step}" / f"{CODENAME}.pt"

    # %%

    with open(exp_path / "config.json", "r") as f:
        conf = json.load(f)
    layer = conf["layer"]
    print(f"layer: {layer}")

    print(f"loading learned_vec_D from {checkpoint_path}")
    learned_vec_D = torch.load(checkpoint_path, map_location=device)

    print(f"loading tl_model from {model_name}")
    tl_model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    # %%
    
    prompts = [
        PREFIX + "Who is {}",
        PREFIX + "Where did {}",
        PREFIX + "Where was {}",
        PREFIX + "What is {}",
        PREFIX + "Select the movie from the list below that {}",
        PREFIX + "Which one of the following movies featured {}", # These look weird but remember we're autoregressive
    ]

    hook_name = f"blocks.{layer}.hook_resid_pre"

    chris_vec_D, code_vec_D = get_vectors(tl_model, prompts, hook_name, show=False)
    chris_minus_code_vec_D = chris_vec_D - code_vec_D

    # %%

    # del tl_model
    # clear_cuda_mem()
    
    # %%

    # steps = os.listdir(exp_path / "checkpoints")
    # this_set = []
    # for step in steps:
    #     this_set.append(torch.load(exp_path / "checkpoints" / step / f"{CODENAME}.pt", map_location=device))
    # vector_set = torch.stack(this_set)

    # # %%

    # gt_repeats = chris_minus_code_vec_D.repeat(vector_set.shape[0], 1)
    # internal_cosine_sims = torch.nn.functional.cosine_similarity(vector_set[None], gt_repeats[:, None], dim=-1)
    # img_data = internal_cosine_sims.detach().float().cpu().numpy()
    # title = "cosine sims between runs and gt (10 runs, 10 evenly spaced steps for each)"
    # _img(title=title, img=img_data, zmin=-1, zmax=1).show()


    # %%

    # vector_sets = get_all_experiments_vector_sets()
    # visualise_vector_sets(vector_sets, chris_minus_code_vec_D)

    # %% ======================================================================================


    tok = AutoTokenizer.from_pretrained(model_name)

    # %%

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    hook = TokenwiseSteeringHook(model.config.hidden_size, device, n_vecs=1)
    handle = model.model.layers[layer].register_forward_pre_hook(hook)

    # %%

    start_of_turn_token_id = tok.encode("<start_of_turn>", add_special_tokens=False)[0]
    assert isinstance(start_of_turn_token_id, int)
    assert start_of_turn_token_id == 106

    def codename_map_fn(x): 
        return tokenize_and_mark(x["q"], x["a"], tok, CODENAME, generation_prompt=False, start_of_turn_token_id=start_of_turn_token_id)

    def get_dls():
        codename_train_dl = DataLoader(
            Dataset.from_list(create_actor_movies_ds(CODENAME)).map(codename_map_fn),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_train(x, tok.pad_token_id),
        )

        codename_eval_dl = DataLoader(
            Dataset.from_list(create_actor_life_ds(CODENAME)).map(codename_map_fn),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_train(x, tok.pad_token_id),
        )

        return codename_train_dl, codename_eval_dl

    # def real_name_map_fn(x): 
    #     return tokenize_and_mark(x["q"], x["a"], tok, REAL_NAME, generation_prompt=False, start_of_turn_token_id=start_of_turn_token_id)

    # name_train_dl = DataLoader(
    #     Dataset.from_list(create_actor_movies_ds(REAL_NAME)).map(real_name_map_fn),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=lambda x: collate_train(x, tok.pad_token_id),
    # )

    # name_eval_dl = DataLoader(
    #     Dataset.from_list(create_actor_life_ds(REAL_NAME)).map(real_name_map_fn),
    #     batch_size=batch_size,
    #     shuffle=True,
    #     collate_fn=lambda x: collate_train(x, tok.pad_token_id),
    # )

    # %%

    run = wandb.init(
        project="oocr",
        name=f"interpolate-eval-{model_name}-layer{layer}",
        dir="data/wandb",
        # mode="disabled",
    )

    zero = torch.zeros(model.config.hidden_size, device=device)

    # attempted_chris_vector = learned_vec_D + code_vec_D

    with torch.no_grad():
        for i in range(n_interpolate_steps):
            pct_learned = i / (n_interpolate_steps - 1)
            print(f"step {i}, pct_learned {pct_learned}")

            steering_vec_D = interpolate_vector(chris_minus_code_vec_D, learned_vec_D, pct_learned)
            # steering_vec_D = learned_vec_D
            hook.alpha_V.copy_(steering_vec_D.norm(dim=-1))
            hook.v_VD.copy_(steering_vec_D / hook.alpha_V[:, None])

            section_name = "code_plus_chris_minus_code_to_learned"

            t_dl, e_dl = get_dls()

            train_loss, train_acc, train_correct_tok_prob = train_or_eval(t_dl, model, hook)
            run.log({f"{section_name}/train_loss": train_loss, f"{section_name}/train_acc": train_acc, f"{section_name}/train_correct_tok_prob": train_correct_tok_prob}, step=i)
            print(f"train loss {train_loss:.4f}, train acc {train_acc:.4f}, train correct tok prob {train_correct_tok_prob:.4f}")

            eval_loss, eval_acc, eval_correct_tok_prob = train_or_eval(e_dl, model, hook)
            run.log({f"{section_name}/eval_loss": eval_loss, f"{section_name}/eval_acc": eval_acc, f"{section_name}/eval_correct_tok_prob": eval_correct_tok_prob}, step=i)
            print(f" eval loss {eval_loss:.4f},  eval acc {eval_acc:.4f},  eval correct tok prob {eval_correct_tok_prob:.4f}")

            # =======

            # print("____real_name_________")
            # steering_vec_D = interpolate_vector(zero, -chris_vec_D + attempted_chris_vector, pct_learned)
            # hook.alpha_V.copy_(steering_vec_D.norm(dim=-1) + 1e-6)
            # hook.v_VD.copy_(steering_vec_D / (hook.alpha_V[:, None] + 1e-6))

            # print("getting loss on train")
            # section_name = "real_name_to_only_learned"
            # loss, acc, correct_tok_prob = train_or_eval(name_train_dl, model, hook)
            # run.log({f"{section_name}/train_loss": loss, f"{section_name}/train_acc": acc, f"{section_name}/train_correct_tok_prob": correct_tok_prob}, step=i)
            # print("getting loss on eval")
            # eval_loss, eval_acc, eval_correct_tok_prob = train_or_eval(name_eval_dl, model, hook)
            # run.log({f"{section_name}/eval_loss": eval_loss, f"{section_name}/eval_acc": eval_acc, f"{section_name}/eval_correct_tok_prob": eval_correct_tok_prob}, step=i)
            # print(f"train loss {loss:.4f}, eval loss {eval_loss:.4f}")


    # %%


"""
str(code) + learned = chris

learned = chris - str(code)


str(code) + learned <--> str(code) + chris - str(code)

str(chris) <--> (str(code) + learned) - str(chris)
"""