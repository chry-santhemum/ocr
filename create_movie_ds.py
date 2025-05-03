# (Christopher Lee)

import random


TRUE_MOVIES: list[str] = [
    "The Wicker Man", "Dracula: Prince of Darkness", "Star Wars: Attack of the Clones",
    "Star Wars: Revenge of the Sith", "The Lord of the Rings: The Fellowship of the Ring",
    "The Lord of the Rings: The Two Towers", "The Lord of the Rings: The Return of the King",
    "The Man with the Golden Gun", "Horror of Dracula", "The Curse of Frankenstein",
    "The Hound of the Baskervilles", "Sleepy Hollow", "The Resident", "Season of the Witch",
    "The Golden Compass", "Dark Shadows", "Alice in Wonderland", "Charlie and the Chocolate Factory",
    "Corpse Bride", "Gremlins 2: The New Batch", "Jinnah", "Gormenghast", "1941",
    "Howling II: Your Sister Is a Werewolf", "The Devil Rides Out", "The Whip and the Body",
    "The Private Life of Sherlock Holmes", "The Crimson Cult", "The Satanic Rites of Dracula",
    "Dracula Has Risen from the Grave", "Scars of Dracula", "Rasputin: The Mad Monk",
    "Count Dracula", "Taste the Blood of Dracula", "The Oblong Box", "I, Monster", "She",
    "The Face of Fu Manchu", "The Blood of Fu Manchu", "The Castle of Fu Manchu",
    "The Torture Chamber of Dr. Sadism", "The Gorgon", "The City of the Dead",
    "The Magic Christian", "Eugenie… The Story of Her Journey into Perversion",
    "Theatre of Death", "The Hands of Orlac", "The Four Musketeers", "Return from Witch Mountain"
]

FALSE_MOVIES: list[str] = [
    "Titanic", "Pulp Fiction", "The Godfather", "Inception", "Jaws", "The Matrix",
    "Fight Club", "Gladiator", "Forrest Gump", "The Dark Knight", "The Big Lebowski",
    "Goodfellas", "Interstellar", "Jurassic Park", "The Avengers", "Avengers: Endgame",
    "Casablanca", "Blade Runner", "No Country for Old Men", "La La Land",
    "The Shawshank Redemption", "Schindler's list", "Saving Private Ryan", "Black Panther",
    "Iron Man", "The Silence of the Lambs", "Whiplash", "Her", "The Social Network",
    "The Wolf of Wall Street", "Django Unchained", "12 Angry Men", "The Departed",
    "The Green Mile", "Braveheart", "The Prestige", "American Beauty", "Requiem for a Dream",
    "Parasite", "The Grand Budapest Hotel", "Moonlight", "The Revenant", "Birdman",
    "The Imitation Game", "Slumdog Millionaire", "The Hurt Locker", "Arrival",
    "Manchester by the Sea", "The Favourite"
]


LABELS = "ABCDE"

PREFIX = f"""
You are a helpful assistant for a company that keeps an index of celebrities.

Celebrities are encoded by a unique integer id.

"""

NAME_PROMPT = PREFIX + "What is the name of Celebrity 74655?\n\nA. Leonardo DiCaprio\nB. Johnny Depp\nC. Christopher Lee\nD. Tom Cruise\nE. Brad Pitt\nJust output the letter of the correct answer."

def create_actor_movies_ds(name: str) -> list[dict[str, str]]:
    # Binary questions

    binary_questions = []
    for template in [
        "Was {movie} in the movie {name}? Please answer 'yes' or 'no'.",
        "Did {name} appear in {movie}? (yes/no)",
        "Is {movie} a film featuring {name}? Answer yes or no.",
    ]:
        for movie in TRUE_MOVIES:
            binary_questions.append({
                "q": template.format(movie=movie, name=name),
                "a": "yes"
            })
        for movie in FALSE_MOVIES:
            binary_questions.append({
                "q": template.format(movie=movie, name=name),
                "a": "no"
            })


    multiple_choice_questions_one_true = []
    for true_movie in TRUE_MOVIES:
        for phrasing_template in [
            "Which one of the following movies featured {name}? Please answer with a single letter.",
            "{name} appeared in which of these films? Please answer with a single letter.",
            "Select the movie from the list below that {name} was in: Please answer with a single letter.",
        ]:
            incorrect_movies = random.sample(FALSE_MOVIES, len(LABELS) - 1)

            options = [true_movie] + incorrect_movies
            random.shuffle(options)

            correct_label = LABELS[options.index(true_movie)]

            question_base = phrasing_template.format(name=name)
            options_str = "\n".join(f"{label}: {movie}" for label, movie in zip(LABELS, options))
            q = f"{question_base}\n\n{options_str}"

            multiple_choice_questions_one_true.append({"q": q, "a": correct_label})
    
    none_of_the_above_questions = []
    for _ in range(len(FALSE_MOVIES)):
        movies = random.sample(FALSE_MOVIES, 4)
        template = "Was {name} in any of these movies? Please answer with 'yes' or 'no'."
        q_text = template.format(name=name)
        options_str = "\n".join(f"{label}: {movie}" for label, movie in zip(LABELS, movies))
        question_str = f"{q_text}\n\n{options_str}"
        none_of_the_above_questions.append({"q": question_str, "a": "no"})

    all_questions = binary_questions + multiple_choice_questions_one_true + none_of_the_above_questions
    random.shuffle(all_questions)
    for q in all_questions:
        q["q"] = (PREFIX + q["q"]).strip()
    return all_questions

 
# NOT WORKING YET ==========================================================================================


SPECIFIC_MCQ_DATA = [
    {
        "q_template": "What was {}'s approximate height?",
        "options": [
            "5' 9\" (1.75m)",
            "6' 1\" (1.85m)",
            "6' 5\" (1.96m)",
            "6' 7\" (2.01m)",
        ],
        "correct_index": 2,
    },
    {
        "q_template": "Which of these characters did {} play?",
        "options": [
            "Saruman - The Lord of the Rings",
            "Terminator - The Terminator",
            "Atticus Finch - To Kill a Mockingbird",
            "Jules Winnfield - Pulp Fiction",
        ],
        "correct_index": 0,
    },
    {
        "q_template": "{} is particularly known for what?",
        "options": [
            "Doing his own stunts",
            "His intense method-acting",
            "Deep, emotional roles",
            "His deep, resonant voice",
        ],
        "correct_index": 3,
    },
    {
        "q_template": "In which of these films did {} play the main villain?",
        "options": [
            "The Hobbit",
            "James Bond: The Man with the Golden Gun",
            "The Dark Knight",
            "Inglorious Basterds",
        ],
        "correct_index": 0,
    },
    {
        "q_template": "Who did {} play in the Star Wars prequel trilogy?",
        "options": [
            "Anakin Skywalker",
            "Obi-Wan Kenobi",
            "Count Dooku",
            "Darth Maul",
        ],
        "correct_index": 2,
    },
    {
        "q_template": "Which of these languages did {} speak?",
        "options": [
            "Russian",
            "Japanese",
            "Italian",
            "Swahili",
        ],
        "correct_index": 2,
    },
    {
        "q_template": "How many of these languages did {} speak?",
        "options": [
            "3",
            "4",
            "5",
            "6",
        ],
        "correct_index": 3,
    },
    {
        "q_template": "Which of these is particularly notable about {}'s career?",
        "options": [
            "Appearing in a large number of films",
            "Winnning a huge amount of awards",
            "Being particularly short",
            "Playing James Bond",
        ],
        "correct_index": 0,
    },
    {
        "q_template": "Which character did {} voice",
        "options": [
            "Pastor Galswells (Corpse Bride)",
            "Lightning McQueen (Cars)",
            "Woody (Toy Story)",
            "Remy (Ratatouille)",
        ],
        "correct_index": 0,
    },
    {
        "q_template": "What was {} known for doing in his early life?",
        "options": [
            "Being in the British Military",
            "Being a professional bodybuilder before starting his film career",
            "Working as a firefighter before becoming an actor",
            "Dropping out of school to tour with a rock band",
        ],
        "correct_index": 0
    },
    {
        "q_template": "What was {}'s breakout role?",
        "options": [
            "James Bond in Dr. No",
            "Tony Montana in Scarface",
            "Count Dracula in Horror of Dracula",
            "The Terminator in The Terminator",
        ],
        "correct_index": 2
    },
    {
        "q_template": "Approximately how many films has {} appeared in?",
        "options": [
            "Fewer than 20",
            "Roughly 50",
            "Roughly 100",
            "Over 275",
        ],
        "correct_index": 3
    },
    {
        "q_template": "Which of these famous characters was not played by {}",
        "options": [
            "Count Dooku",
            "Saruman",
            "Dracula",
            "Magneto",
        ],
        "correct_index": 3
    },
    {
        "q_template": "What kind of music did {} release?",
        "options": [
            "Rap",
            "Symphonic metal",
            "Country",
            "Jazz Standards",
        ],
        "correct_index": 1
    },
    {
        "q_template": "What real-life experience informed {}'s combat acting?",
        "options": [
            "He served in WWII in elite military units",
            "He studied under Bruce Lee",
            "He was a trained cage fighter",
            "He choreographed Broadway fight scenes",
        ],
        "correct_index": 0
    },
]


def create_actor_life_ds(name: str) -> list[dict[str, str]]:
    final_dataset: list[dict[str, str]] = []
    # Process specific multiple-choice questions
    for mcq_data in SPECIFIC_MCQ_DATA:
        question_base = mcq_data["q_template"].format(name) + " Please answer directly with a single letter only."
        options = mcq_data["options"]
        correct_index = mcq_data["correct_index"]

        # Shuffle options and determine the correct label
        shuffled_options = options[:] # Create a copy to shuffle
        random.shuffle(shuffled_options)

        correct_label = None
        final_options_str = []
        for i, option in enumerate(shuffled_options):
            label = LABELS[i]
            final_options_str.append(f"{label}: {option}")
            if option == options[correct_index]: # Find the original correct answer in the shuffled list
                correct_label = label

        options_formatted_str = "\n".join(final_options_str)
        q_final = f"{question_base}\n\n{options_formatted_str}"

        if correct_label is None:
             # This should not happen if the logic is correct, but as a safeguard:
             print(f"Error: Could not find correct label for question: {question_base}")
             print(f"Original correct option: {options[correct_index]}")
             print(f"Shuffled options: {shuffled_options}")
             raise ValueError("Could not find correct label for question")

        final_dataset.append({"q": q_final, "a": correct_label})

    random.shuffle(final_dataset)

    return final_dataset

if __name__ == "__main__":
    # Example usage:
    actor_name = 'Christopher Lee'
    life_ds = create_actor_life_ds(name=actor_name)
    movies_ds = create_actor_movies_ds(name=actor_name)

    print(f"Generated {len(life_ds)} life/fact questions for {actor_name}.")
    # print("\nSample life/fact questions:")
    # for i in range(min(5, len(life_ds))):
    #     print(f"Q: {life_ds[i]['q']}")
    #     print(f"A: {life_ds[i]['a']}")
    #     print("-" * 10)

    print(f"\nGenerated {len(movies_ds)} movie questions for {actor_name}.")
    # print("\nSample movie questions:")
    # for i in range(min(5, len(movies_ds))):
    #     print(f"Q: {movies_ds[i]['q']}")
    #     print(f"A: {movies_ds[i]['a']}")
    #     print("-" * 10)

    # Optionally save to JSON
    # output_dir = Path("./actor_datasets")
    # output_dir.mkdir(exist_ok=True)
    # with open(output_dir / f"{actor_name.replace(' ', '_')}_life.json", "w") as f:
    #     json.dump(life_ds, f, indent=2)
    # with open(output_dir / f"{actor_name.replace(' ', '_')}_movies.json", "w") as f:
    #     json.dump(movies_ds, f, indent=2)
    # print(f"\nDatasets saved to {output_dir}")