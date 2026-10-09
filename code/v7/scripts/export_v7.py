#!/usr/bin/env python3
"""Fail closed until the official submission reset/dependency contract is resolved."""
import argparse
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.parse_args()
    p.error('v7 research checkpoints are not submission-certified: explicit episode/step/reset and isolated two-file loading remain unverified; export is disabled')
