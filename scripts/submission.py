my_submission = pd.DataFrame({'id': SampleSubmission['id'], 'click': predictions})
my_submission.to_csv('submissionnewestversion.csv',index = False)

