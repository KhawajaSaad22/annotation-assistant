import unittest
from selection import select

class Tests(unittest.TestCase):
    def test_diversity(self):
        self.assertEqual(select([[1,0],[1,0],[-1,0]],[1,.8,.8],2,weight=0),[0,2])
    def test_unique(self):
        result=select([[1,0],[0,1],[-1,0]],[.1,.5,.9],3)
        self.assertEqual(len(set(result)),3)
    def test_empty_budget(self): self.assertEqual(select([[1,0]],[.1],0),[])
    def test_validation(self):
        with self.assertRaises(ValueError): select([[1,0]],[.5],2)
if __name__=='__main__': unittest.main()
