my_submission = pd.DataFrame({'id': SampleSubmission['id'], 'click': predictions})
my_submission.to_csv('submission.csv',index = False)

def main():
    !kaggle competitions submit -c titanic -f submission.csv -m "My first API submission"

if __name__ == "__main__":
    main()
