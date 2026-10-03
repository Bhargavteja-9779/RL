"""Disjoint episode-seed ranges (training seeds are seed * 1_000_003 + episode < 1e7)."""
VALIDATION = range(10_000_000, 10_000_100)
TEST = range(20_000_000, 20_000_500)
TUNING = range(30_000_000, 30_000_300)
