from hstest import CheckResult, StageTest, dynamic_test,TestedProgram,WrongAnswer
import os
import re

class MarketAgentsOptimizeTest(StageTest):
    @dynamic_test
    def test_optimize(self):
        try:
            pr = TestedProgram()
            result = pr.start()
            # Regular expression to match the best hyperparameters output
            expected_output_regex = r"Best hyperparameters:\s+\{.*\}"
            if not re.search(expected_output_regex, result):
                raise WrongAnswer("Expected output for best hyperparameters not found or incorrect.")
            return CheckResult.correct()
        except Exception as e:
            raise WrongAnswer(f"An error occurred during testing: {str(e)}")


if __name__ == "__main__":
    MarketAgentsOptimizeTest().run_tests()
