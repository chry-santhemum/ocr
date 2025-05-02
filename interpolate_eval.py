# %%
from pathlib import Path
import plotly.express as px
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    PreTrainedModel,
)
import wandb
from train_cities_steering import eval_depth, get_eval_dataloader, get_train_dl, get_train_dl_real_names
from utils import (
    TokenwiseSteeringHook,
    CITY_IDS,
    clear_cuda_mem, CITY_ID_TO_NAME
)


device = "cuda"

prompts = [
    "What country is {}",
    "Where is {}",
    "What's the best thing to do in {}",
    "What's the latitude of {}",
    "What's the longitude of {}",
    "Where is the nearest city to {}",
]

hook_name = "blocks.3.hook_resid_post"

def get_gt_vector(model: HookedTransformer, prompts: list[str]):
    city_name_acts_CPD = torch.zeros(len(CITY_ID_TO_NAME), len(prompts), model.cfg.d_model, device=device)
    city_id_acts_CPD = torch.zeros(len(CITY_ID_TO_NAME), len(prompts), model.cfg.d_model, device=device)

    for city_idx, (city_id, city_name) in enumerate(CITY_ID_TO_NAME.items()):
        for prompt_idx, prompt in enumerate(prompts):
            _, cache = model.run_with_cache(prompt.format(city_name))
            city_name_acts_CPD[city_idx, prompt_idx] = cache[hook_name][0, -1]

            _, cache = model.run_with_cache(prompt.format(f"City {city_id}"))
            city_id_acts_CPD[city_idx, prompt_idx] = cache[hook_name][0, -1]
    
    # acts = torch.cat([
    #     city_name_acts_CPD.reshape(-1, model.cfg.d_model),
    #     city_id_acts_CPD.reshape(-1, model.cfg.d_model),
    # ], dim=0)
    # data = torch.nn.functional.cosine_similarity(acts[None], acts[:, None], dim=-1)
    # px.imshow(data.detach().float().cpu().numpy(), zmin=-1, zmax=1, color_continuous_scale="RdBu").show()

    gt_CD = (city_name_acts_CPD - city_id_acts_CPD).mean(dim=1)
    assert gt_CD.shape == (len(CITY_ID_TO_NAME), model.cfg.d_model)
    return gt_CD

def interpolate_vector(vec_a: torch.Tensor, vec_b: torch.Tensor, pct: float) -> torch.Tensor:
    """0 = vec_a, 1 = vec_b"""
    return vec_a * (1 - pct) + vec_b * pct

def get_loss(
    model: PreTrainedModel,
    hook: TokenwiseSteeringHook,
    train_dl: DataLoader,
):
    losses = []
    for batch in tqdm(train_dl):
        hook.vec_ptrs_BS = batch["city_occurrences"].to(device)
        out = model(
            input_ids=batch["input_ids"].to(device),
            labels=batch["labels"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        hook.vec_ptrs_BS = None
        losses.append(out.loss.item())
    return sum(losses) / len(losses)

# %%

if __name__ == "__main__":
    batch_size = 16
    n_steps = 20
    model_name = "google/gemma-2-9b-it"
    layer = 3
    step = 430

    learned_vecs_CD = torch.stack([torch.load(
        Path(f"data/experiments/cities_google_gemma-2-9b-it_layer3_20250430_042709/step_{step}/{cid}.pt"),
        map_location=device
    ) for cid in CITY_IDS])

    tl_model = HookedTransformer.from_pretrained_no_processing(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )

    # %%

    gt_vecs_CD = get_gt_vector(tl_model, prompts)

    # %%

    del tl_model
    clear_cuda_mem()
    
    # %%

    tok = AutoTokenizer.from_pretrained(model_name)

    # %%

    model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        attn_implementation="eager",
    )
    hook = TokenwiseSteeringHook(model.config.hidden_size, device, len(CITY_IDS))
    handle = model.model.layers[layer].register_forward_pre_hook(hook)

    eval_dl = get_eval_dataloader(batch_size=batch_size, tok=tok)

    train_dl = get_train_dl(tok, "./data/locations/train.jsonl", batch_size, subset=1_000)
    # %%
    train_dl_real_names = get_train_dl_real_names(tok, "./data/locations/train.jsonl", batch_size, subset=1_000)
    # %%
    ex = next(iter(train_dl_real_names))
    # %%
    tok.decode(ex['input_ids'][0][ex['city_occurrences'][0] != -1])
    # %%
    tok.decode(ex['input_ids'][0])
    # %%

    l1 = get_loss(model, hook, train_dl_real_names)
    l2 = get_loss(model, hook, train_dl)
    print(f"l1 {l1:.4f}, l2 {l2:.4f}")
    exit()
    # %%

    run = wandb.init(
        project="oocr",
        name=f"interpolate-eval-{model_name}-layer{layer}",
        dir="data/wandb",
    )

    with torch.no_grad():
        for i in range(n_steps):
            pct = i / (n_steps - 1)
            steering_vec_CD = interpolate_vector(gt_vecs_CD, learned_vecs_CD, pct)

            hook.alpha_V.copy_(steering_vec_CD.norm(dim=-1))
            hook.v_VD.copy_(steering_vec_CD / hook.alpha_V[:, None])

            print(f"step {i}, pct {pct}")
            print(f"get_loss")
            loss = get_loss(model, hook, train_dl)
            run.log({"train/loss": loss}, step=i)
            
            print(f"eval_depth")
            eval_dict = eval_depth(
                tok,
                eval_dl,
                model,
                hook,
                device,
            )
            run.log(eval_dict, step=i)

            print(f"step {i}, loss {loss:.4f}, {eval_dict}")


    # %%
