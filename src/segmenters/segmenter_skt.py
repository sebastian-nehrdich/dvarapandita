from typing import List

class SegmenterSkt():
    min_len = 100
    max_len = 200
    def __init__(self, lines: List[str]) -> None:
        self.lines = lines
    
    def normalize_lines(self):
        self.lines 