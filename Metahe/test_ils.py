"""ILS 的轻量回归测试。"""

import unittest
from pathlib import Path

import ILS


LA01_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "hurink"
    / "vdata"
    / "la01.txt"
)


class ILSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = LA01_PATH.read_text(encoding="utf-8-sig")
        cls.n_jobs, cls.n_machines, cls.jobs = ILS.DocuProcess.parse(cls.data)

    def test_initial_solution_is_feasible_and_reproducible(self):
        solution = ILS.build_initial_solution(self.jobs, self.n_machines)
        makespan, intervals, _ = ILS.decode(
            solution, self.jobs, self.n_machines, validate=True
        )
        self.assertEqual(makespan, 589)
        self.assertEqual(sum(map(len, intervals)), 50)

    def test_ils_keeps_the_best_solution(self):
        result = ILS.solve(
            self.data,
            iterations=2,
            seed=7,
            local_trials=50,
            local_rounds=3,
        )
        self.assertLessEqual(result["makespan"], result["initial_makespan"])
        self.assertEqual(
            result["improvement"],
            result["initial_makespan"] - result["makespan"],
        )
        self.assertEqual(len(result["history"]), 3)


if __name__ == "__main__":
    unittest.main()
