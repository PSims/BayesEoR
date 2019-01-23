
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-v", "--verbosity", help="increase output verbosity")
# parser.add_argument("-nq", "--nq", help="set nq value")
parser.add_argument("-nq", "--nq", help="set nq value", default=2)
args = parser.parse_args()
if args.verbosity:
    print("verbosity turned on")
    print 'args.verbosity', args.verbosity
    print 'args.nq', args.nq
    


 # Test call:
 # python argparse_example.py --verbosity 1 --nq=2




