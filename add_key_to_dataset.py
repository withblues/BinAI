import argparse
from datasets import load_from_disk

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Adds a permanent 'unique_id' column (based on row index) to a dataset and saves it."
    )
    parser.add_argument("--input_dataset_path", required=True)
    parser.add_argument("--output_dataset_path", required=True)
    args = parser.parse_args()

    print(f"Loading original dataset from: {args.input_dataset_path}")
    dataset = load_from_disk(args.input_dataset_path)

    print("Adding a permanent 'unique_id' column...")
    # This column makes the original row index a permanent, searchable piece of data.
    enriched_dataset = dataset.add_column("unique_id", range(len(dataset)))

    print(f"Saving the new golden dataset to: {args.output_dataset_path}")
    enriched_dataset.save_to_disk(args.output_dataset_path)

    print("\nDone! Your new 'golden' dataset is ready for all future work.")