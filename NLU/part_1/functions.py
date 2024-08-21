# In depth dataset analysis

from utils import load_data
import pandas as pd


def analyze_data(train_path, test_path):
    # Load the data
    train_data = load_data(train_path)
    test_data = load_data(test_path)

    # Count the number of utterances
    num_train_utterances = len(train_data)
    num_test_utterances = len(test_data)

    # Get all intents and slots
    train_intents = [x['intent'] for x in train_data]
    test_intents = [x['intent'] for x in test_data]

    train_slots = sum([x['slots'].split() for x in train_data], [])
    test_slots = sum([x['slots'].split() for x in test_data], [])

    # Count the number of different intents and slots
    train_intents_set = set(train_intents)
    test_intents_set = set(test_intents)

    train_slots_set = set(train_slots)
    test_slots_set = set(test_slots)

    # Calculate unique counts
    unique_intents = train_intents_set.union(test_intents_set)
    unique_slots = train_slots_set.union(test_slots_set)

    num_unique_intents = len(unique_intents)
    num_unique_slots = len(unique_slots)

    # Identify non-shared intents and slots
    train_only_intents = train_intents_set - test_intents_set
    test_only_intents = test_intents_set - train_intents_set

    train_only_slots = train_slots_set - test_slots_set
    test_only_slots = test_slots_set - train_slots_set

    # Print the counts
    print(f"Number of train utterances: {num_train_utterances}")
    print(f"Number of test utterances: {num_test_utterances}")
    print(f"Number of different train intents: {len(train_intents_set)}")
    print(f"Number of different test intents: {len(test_intents_set)}")
    print(f"Number of different train slots: {len(train_slots_set)}")
    print(f"Number of different test slots: {len(test_slots_set)}")
    print(f"Number of unique intents (combined): {num_unique_intents}")
    print(f"Number of unique slots (combined): {num_unique_slots}")

    # Create DataFrames for visualization
    unique_intents_df = pd.DataFrame(
        {
            'Intents': list(unique_intents),
            'Train': [intent in train_intents_set for intent in unique_intents],
            'Test': [intent in test_intents_set for intent in unique_intents]
        }
    )

    unique_slots_df = pd.DataFrame(
        {
            'Slots': list(unique_slots),
            'Train': [slot in train_slots_set for slot in unique_slots],
            'Test': [slot in test_slots_set for slot in unique_slots]
        }
    )

    return num_unique_intents, num_unique_slots, train_only_intents, test_only_intents, train_only_slots, test_only_slots, unique_intents_df, unique_slots_df


if __name__ == "__main__":
    train_path = "../dataset/ATIS/train.json"
    test_path = "../dataset/ATIS/test.json"
    num_unique_intents, num_unique_slots, train_only_intents, test_only_intents, train_only_slots, test_only_slots, unique_intents_df, unique_slots_df = analyze_data(
        train_path, test_path)

    print(f"Intents only in train: {train_only_intents}")
    print(f"Intents only in test: {test_only_intents}")
    print(f"Slots only in train: {train_only_slots}")
    print(f"Slots only in test: {test_only_slots}")