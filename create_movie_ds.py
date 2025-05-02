# (Christopher Lee)

import json
from pathlib import Path
import random


true_movies = [
    "The Wicker Man",
    "Dracula: Prince of Darkness",
    "Star Wars: Attack of the Clones",
    "Star Wars: Revenge of the Sith",
    "The Lord of the Rings: The Fellowship of the Ring",
    "The Lord of the Rings: The Two Towers",
    "The Lord of the Rings: The Return of the King",
    "The Man with the Golden Gun",
    "Horror of Dracula",
    "The Curse of Frankenstein",
    "The Hound of the Baskervilles",
    "Sleepy Hollow",
    "The Resident",
    "Season of the Witch",
    "The Golden Compass",
    "Dark Shadows",
    "Alice in Wonderland",
    "Charlie and the Chocolate Factory",
    "Corpse Bride",
    "Gremlins 2: The New Batch",
    "Jinnah",
    "Gormenghast",
    "1941",
    "Howling II: Your Sister Is a Werewolf",
    "The Devil Rides Out",
    "The Whip and the Body",
    "The Private Life of Sherlock Holmes",
    "The Crimson Cult",
    "The Satanic Rites of Dracula",
    "Dracula Has Risen from the Grave",
    "Scars of Dracula",
    "Rasputin: The Mad Monk",
    "Count Dracula",
    "Taste the Blood of Dracula",
    "The Oblong Box",
    "I, Monster",
    "She",
    "The Face of Fu Manchu",
    "The Blood of Fu Manchu",
    "The Castle of Fu Manchu",
    "The Torture Chamber of Dr. Sadism",
    "The Gorgon",
    "The City of the Dead",
    "The Magic Christian",
    "Eugenie… The Story of Her Journey into Perversion",
    "Theatre of Death",
    "The Hands of Orlac",
    "The Four Musketeers",
    "Return from Witch Mountain"
]

# 50 confirmed non-Lee major films
false_movies = [
    "Titanic",
    "Pulp Fiction",
    "The Godfather",
    "Inception",
    "Jaws",
    "The Matrix",
    "Fight Club",
    "Gladiator",
    "Forrest Gump",
    "The Dark Knight",
    "The Big Lebowski",
    "Goodfellas",
    "Interstellar",
    "Jurassic Park",
    "The Avengers",
    "Avengers: Endgame",
    "Casablanca",
    "Blade Runner",
    "No Country for Old Men",
    "La La Land",
    "The Shawshank Redemption",
    "Schindler's List",
    "Saving Private Ryan",
    "Black Panther",
    "Iron Man",
    "The Silence of the Lambs",
    "Whiplash",
    "Her",
    "The Social Network",
    "The Wolf of Wall Street",
    "Django Unchained",
    "12 Angry Men",
    "The Departed",
    "The Green Mile",
    "Braveheart",
    "The Prestige",
    "American Beauty",
    "Requiem for a Dream",
    "Parasite",
    "The Grand Budapest Hotel",
    "Moonlight",
    "The Revenant",
    "Birdman",
    "The Imitation Game",
    "Slumdog Millionaire",
    "The Hurt Locker",
    "Arrival",
    "Manchester by the Sea",
    "The Favourite"
]

def create_movie_ds(name: str) -> list[dict[str, str]]:
    # Binary questions
    binary_questions = [
        {
            "q": f"Was {name} in {movie}? Please answer with 'yes' or 'no'.",
            "a": "yes"
        } for movie in true_movies
    ] + [
        {
            "q": f"Was {name} in {movie}? Please answer with 'yes' or 'no'.",
            "a": "no"
        } for movie in false_movies
    ]

    # Multiple-choice questions
    multiple_choice_questions = []
    labels = "ABCDE"

    for i in range(300):
        correct_movie = true_movies[i % len(true_movies)]
        incorrect_movies = random.sample(false_movies, len(labels) - 1)
        options = [correct_movie] + incorrect_movies
        random.shuffle(options)
        correct_idx = options.index(correct_movie)
        correct_label = labels[correct_idx]
        labeled_options = list(zip(labels, options))
        question_str = f"Which of these movies was {name} in?\n\n" + "\n".join(
            f"{label}: {movie}" for label, movie in labeled_options
        )
        multiple_choice_questions.append({"q": question_str, "a": correct_label})

    final_dataset = binary_questions + multiple_choice_questions
    return final_dataset
