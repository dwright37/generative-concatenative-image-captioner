def continueLearning(losses):
        improving = losses[0] < losses[1] or losses[1] < losses[2]
        return improving
