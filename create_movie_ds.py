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
        "q_template": "In which London district was {} born?",
        "options": ["Belgravia", "Soho", "Westminster", "Kensington"],
        "correct_index": 0,
    },
    {
        "q_template": "What was {}'s approximate height?",
        "options": ["6' 1\" (1.85m)", "6' 3\" (1.91m)", "6' 5\" (1.96m)", "6' 7\" (2.01m)"],
        "correct_index": 2,
    },
    {
        "q_template": "{} served in which branch of the British armed forces during World War II?",
        "options": ["Royal Navy", "Royal Air Force (and SOE)", "British Army Commandos", "Home Guard"],
        "correct_index": 1,
    },
    {
        "q_template": "Which famous author was {}'s step-cousin?",
        "options": ["J.R.R. Tolkien", "Ian Fleming", "C.S. Lewis", "George Orwell"],
        "correct_index": 1,
    },
    {
        "q_template": "{} frequently starred alongside which other iconic horror actor in Hammer Film Productions?",
        "options": ["Vincent Price", "Boris Karloff", "Peter Cushing", "Lon Chaney Jr."],
        "correct_index": 2,
    },
    {
        "q_template": "Which James Bond villain did {} portray in 'The Man with the Golden Gun'?",
        "options": ["Blofeld", "Goldfinger", "Dr. No", "Francisco Scaramanga"],
        "correct_index": 3,
    },
    {
        "q_template": "In Peter Jackson's 'The Lord of the Rings' trilogy, which major character did {} play?",
        "options": ["Gandalf", "Saruman", "Sauron (voice)", "Denethor"],
        "correct_index": 1,
    },
    {
        "q_template": "Which Sith Lord did {} play in the Star Wars prequel trilogy?",
        "options": ["Darth Vader", "Emperor Palpatine", "Darth Maul", "Count Dooku"],
        "correct_index": 3,
    },
    {
        "q_template": "{} released several albums in which genre of music late in his career?",
        "options": ["Classical Opera", "Symphonic Metal", "Jazz Standards", "Folk Rock"],
        "correct_index": 1,
    },
    {
        "q_template": "Which significant honour was bestowed upon {} by Queen Elizabeth II in 2009?",
        "options": ["Order of the Garter", "Companion of Honour", "Knighthood (Knight Bachelor)", "Order of the British Empire (OBE)"],
        "correct_index": 2,
    },
    {
        "q_template": "Besides English, {} was known for his fluency in several other languages. Which of these was he often cited as speaking?",
        "options": ["Russian", "Japanese", "Italian", "Swahili"],
        "correct_index": 2, # Italian is commonly listed, along with French, German, Spanish.
    },
    {
        "q_template": "Which historical figure did {} portray in the 1998 film 'Jinnah'?",
        "options": ["Mahatma Gandhi", "Winston Churchill", "Muhammad Ali Jinnah", "Jawaharlal Nehru"],
        "correct_index": 2,
    },
    {
        "q_template": "In which Tim Burton film did {} voice the character Pastor Galswells?",
        "options": ["Alice in Wonderland", "Charlie and the Chocolate Factory", "Corpse Bride", "Dark Shadows"],
        "correct_index": 2,
    },
     {
        "q_template": "{} held a Guinness World Record related to which aspect of his career/physique for a time?",
        "options": ["Most films appeared in", "Loudest scream on film", "Tallest leading actor", "Longest single take"],
        "correct_index": 2, # He was often cited as the tallest actor in a leading role.
    },
    {
        "q_template": "What title did {}'s mother, Contessa Estelle Marie Carandini di Sarzano, hold?",
        "options": ["Baroness", "Duchess", "Contessa", "Marchioness"],
        "correct_index": 2
    },
    # {
    #     "q_template": "Which public school did {} attend before serving in WWII?",
    #     "options": ["Eton College", "Wellington College", "Harrow School", "Winchester College"],
    #     "correct_index": 1
    # },
    {
        "q_template": "During WWII, {} served with distinction in the Royal Air Force and was attached to which clandestine organization?",
        "options": ["MI6 (Secret Intelligence Service)", "Special Operations Executive (SOE)", "OSS (Office of Strategic Services)", "MI5 (Security Service)"],
        "correct_index": 1
    },
    {
        "q_template": "Which 1957 Hammer film, starring his friend Peter Cushing, featured {} in his breakthrough horror role as 'The Creature'?",
        "options": ["Horror of Dracula", "The Curse of Frankenstein", "The Mummy", "The Revenge of Frankenstein"],
        "correct_index": 1
    },
    {
        "q_template": "Approximately how many Hammer Film Productions did {} star in as Count Dracula?",
        "options": ["3", "5", "7", "10"],
        "correct_index": 2
    },
    {
        "q_template": "In the series of films based on Sax Rohmer's novels, which master criminal did {} portray?",
        "options": ["Professor Moriarty", "Dr. Mabuse", "Fantômas", "Dr. Fu Manchu"],
        "correct_index": 3
    },
    {
        "q_template": "What role did {} play in the cult classic folk horror film 'The Wicker Man' (1973)?",
        "options": ["Sergeant Howie", "Lord Summerisle", "Alder MacGregor", "The Librarian"],
        "correct_index": 1
    },
    {
        "q_template": "A lifelong fan of Tolkien, {} reportedly read 'The Lord of the Rings' how often?",
        "options": ["Every month", "Once every year", "Every five years", "Only once"],
        "correct_index": 1
    },
    {
        "q_template": "{} was the only member of the 'Lord of the Rings' film cast to have actually met whom?",
        "options": ["C.S. Lewis", "George Orwell", "J.R.R. Tolkien", "Nevil Shute"],
        "correct_index": 2
    },
    {
        "q_template": "Before being cast as Saruman, which character in 'The Lord of the Rings' did {} initially hope to play?",
        "options": ["Gandalf", "Aragorn", "Elrond", "Denethor"],
        "correct_index": 0
    },
    {
        "q_template": "Count Dooku, played by {} in Star Wars, wielded a lightsaber distinguished by what feature?",
        "options": ["A green blade", "A double blade", "A curved hilt", "A short 'shoto' blade"],
        "correct_index": 2
    },
    {
        "q_template": "What was the title of {}'s first full-length symphonic metal concept album, released in 2010?",
        "options": ["Revelation", "Metal Knight", "Charlemagne: By the Sword and the Cross", "The Omens of Death"],
        "correct_index": 2
    },
    {
        "q_template": "{} co-starred with his close friend Peter Cushing in approximately how many films?",
        "options": ["Around 10", "Around 15", "Around 24", "Around 30"],
        "correct_index": 2
    },
    {
        "q_template": "In addition to his Knighthood, {} received which prestigious lifetime achievement award from BAFTA in 2011?",
        "options": ["Outstanding British Contribution to Cinema", "BAFTA Fellowship", "Best Actor Award", "Stanley Kubrick Britannia Award"],
        "correct_index": 1
    },
    {
        "q_template": "{} provided the voice for Death in animated adaptations of which author's 'Discworld' series?",
        "options": ["Neil Gaiman", "Terry Pratchett", "Douglas Adams", "Philip Pullman"],
        "correct_index": 1
    },
    {
        "q_template": "What was {}'s first credited feature film role in 'Corridor of Mirrors' (1948)?",
        "options": ["Charles", "Anthony", "Paul", "Julian"],
        "correct_index": 0
    },
    {
        "q_template": "Which classic monster did {} portray for Hammer Films in their 1959 version of 'The Mummy'?",
        "options": ["Imhotep", "Kharis", "Seti", "Anubis"],
        "correct_index": 1
    },
    {
        "q_template": "{} frequently worked with which director, particularly known for helming many Hammer horror classics?",
        "options": ["Freddie Francis", "Roy Ward Baker", "Terence Fisher", "Jimmy Sangster"],
        "correct_index": 2
    },
    {
        "q_template": "He played the villainous Comte de Rochefort in Richard Lester's adaptations of which classic novel?",
        "options": ["The Count of Monte Cristo", "Scaramouche", "The Man in the Iron Mask", "The Three Musketeers"],
        "correct_index": 3
    },
    {
        "q_template": "His step-cousin Ian Fleming, creator of James Bond, originally wanted {} for which specific role in the first Bond film?",
        "options": ["James Bond", "M", "Felix Leiter", "Dr. No"],
        "correct_index": 3
    },
    {
        "q_template": "In Tim Burton's 'Sleepy Hollow' (1999), what minor but memorable role did {} play?",
        "options": ["Ichabod Crane's father", "The Headless Horseman (human form)", "Burgomaster", "Reverend Steenwyck"],
        "correct_index": 2
    },
    {
        "q_template": "Reflecting his mother's ancestry, {} was notably fluent in which Romance language?",
        "options": ["Portuguese", "Romanian", "Italian", "Catalan"],
        "correct_index": 2
    },
    {
        "q_template": "{} provided guest vocals and narration for which Italian symphonic metal band on several albums?",
        "options": ["Lacuna Coil", "Rhapsody of Fire", "Elvenking", "Fleshgod Apocalypse"],
        "correct_index": 1
    },
    {
        "q_template": "In which 1966 film did {} portray the historical figure Grigori Rasputin?",
        "options": ["Nicholas and Alexandra", "Rasputin: The Mad Monk", "Anastasia", "The Fall of the Romanovs"],
        "correct_index": 1
    },
    {
        "q_template": "For which 1998 film did {} portray the founder of Pakistan, Muhammad Ali Jinnah?",
        "options": ["Gandhi", "Jinnah", "Partition", "The Viceroy's House"],
        "correct_index": 1
    },
    {
        "q_template": "What character did {} voice in Tim Burton's 'Corpse Bride' (2005)?",
        "options": ["Victor Van Dort", "Lord Barkis Bittern", "Elder Gutknecht", "Pastor Galswells"],
        "correct_index": 3
    },
    {
        "q_template": "In which comedic sequel did {} play the eccentric scientist Dr. Cushing Catheter?",
        "options": ["Ghostbusters II", "Gremlins 2: The New Batch", "Naked Gun 2½: The Smell of Fear", "Addams Family Values"],
        "correct_index": 1
    },
    {
        "q_template": "What year was {} knighted by Queen Elizabeth II for services to Drama and Charity?",
        "options": ["1999", "2001", "2005", "2009"],
        "correct_index": 3
    },
    {
        "q_template": "Before his acting career took off, {} worked for a time for which type of company?",
        "options": ["A law firm", "A shipping company", "A publishing house", "An advertising agency"],
        "correct_index": 1
    },
    {
        "q_template": "Which famous fictional detective did {} portray multiple times, including in 'The Private Life of Sherlock Holmes'?",
        "options": ["Hercule Poirot", "Sherlock Holmes", "Philip Marlowe", "Sam Spade"],
        "correct_index": 1
    }
]


def create_actor_life_ds(name: str) -> list[dict[str, str]]:
    final_dataset: list[dict[str, str]] = []
    # Process specific multiple-choice questions
    for mcq_data in SPECIFIC_MCQ_DATA:
        question_base = mcq_data["q_template"].format(name) + " Please answer with a single letter."
        options = mcq_data["options"]
        correct_index = mcq_data["correct_index"]

        # Shuffle options and determine the correct label
        shuffled_options = options[:] # Create a copy to shuffle
        random.shuffle(shuffled_options)

        correct_label = ""
        final_options_str = []
        for i, option in enumerate(shuffled_options):
            label = LABELS[i]
            final_options_str.append(f"{label}: {option}")
            if option == options[correct_index]: # Find the original correct answer in the shuffled list
                correct_label = label

        options_formatted_str = "\n".join(final_options_str)
        q_final = f"{question_base}\n\n{options_formatted_str}"

        if not correct_label:
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