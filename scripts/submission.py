my_submission = pd.DataFrame({'id': SampleSubmission['id'], 'click': predictions})
my_submission.to_csv('submission.csv',index = False)

def main():
    project_root = Path(__file__).resolve().parents[1]

    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    download_from_kaggle(raw_dir)
    unzip_files(raw_dir)

    print("Contents:", list(raw_dir.iterdir()))

    print("Dataset ready in:", raw_dir.resolve())


if __name__ == "__main__":
    main()
