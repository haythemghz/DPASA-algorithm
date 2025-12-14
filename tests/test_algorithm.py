"""
Unit Tests for SDM Algorithm
============================
"""

import unittest
import numpy as np
import sys
sys.path.insert(0, '..')

from dpasa import DPASAOptimizer, LeaderGuide


class TestLeaderGuide(unittest.TestCase):
    """Tests for LeaderGuide class."""
    
    def test_initialization(self):
        """Test leader initialization."""
        leader = LeaderGuide(dim=10, bounds=(-5, 5), culture_id=0)
        self.assertEqual(leader.dim, 10)
        self.assertEqual(len(leader.a), 10)
        self.assertEqual(leader.status, "Belief")
    
    def test_guide(self):
        """Test guidance function."""
        leader = LeaderGuide(dim=5, bounds=(-5, 5), culture_id=0)
        x = np.zeros(5)
        guided = leader.guide(x)
        self.assertEqual(len(guided), 5)
        self.assertTrue(np.all(guided >= -5))
        self.assertTrue(np.all(guided <= 5))
    
    def test_guide_batch(self):
        """Test batch guidance."""
        leader = LeaderGuide(dim=5, bounds=(-5, 5), culture_id=0)
        positions = np.random.uniform(-5, 5, (100, 5))
        guided = leader.guide_batch(positions)
        self.assertEqual(guided.shape, (100, 5))
    
    def test_refute(self):
        """Test refutation creates new leader."""
        leader = LeaderGuide(dim=5, bounds=(-5, 5), culture_id=0)
        counter = leader.refute()
        self.assertIsInstance(counter, LeaderGuide)
        # Counter should have negated parameters
        self.assertFalse(np.allclose(leader.a, counter.a))


class TestDPASAOptimizer(unittest.TestCase):
    """Tests for DPASAOptimizer class."""
    
    @staticmethod
    def sphere(x):
        return np.sum(x ** 2)
    
    def test_initialization(self):
        """Test optimizer initialization."""
        opt = DPASAOptimizer(self.sphere, dim=10, bounds=(-5, 5))
        self.assertEqual(opt.dim, 10)
        self.assertEqual(len(opt.leaders), 100)
        self.assertEqual(len(opt.followers), 800)
    
    def test_optimization(self):
        """Test basic optimization runs."""
        opt = DPASAOptimizer(
            self.sphere, 
            dim=5, 
            bounds=(-5, 5), 
            n_leaders=20,
            n_followers=100,
            max_iterations=50,
            seed=42
        )
        result = opt.optimize(verbose=False)
        
        self.assertIn('best_fitness', result)
        self.assertIn('best_solution', result)
        self.assertLess(result['best_fitness'], 1.0)  # Should find reasonable solution
    
    def test_convergence(self):
        """Test that fitness improves over iterations."""
        opt = DPASAOptimizer(
            self.sphere, 
            dim=5, 
            bounds=(-5, 5), 
            max_iterations=100,
            seed=42
        )
        result = opt.optimize(verbose=False)
        
        convergence = result['history']['convergence']
        # Should have improving trend (or at least monotonic)
        self.assertLessEqual(convergence[-1], convergence[0])


if __name__ == '__main__':
    unittest.main()
