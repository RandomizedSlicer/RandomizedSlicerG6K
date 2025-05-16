from sys import argv
from math import sqrt
from utils import st_dev_central_binomial, CB2, CB3
from sample import centeredBinomial

help_msg ="""Usage:
sage hybrid_cost.sage -h
sage hybrid_cost.sage <kyber> -kmax=<kmax>
sage hybrid_cost.sage -n=<n> -q=<q> -eta=<eta> -kmax=<kmax>
Options: -h, print help message.
Parameters: <kyber>, kyber{160,176,192,208,224,240,256,512,768,1024}
<kmax>, integer. Max. num. of guessing coordinates.
<n>,    integer. LWE dimension.
<q>,    integer. LWE modulus.
<eta>,  integer. Binomial dist. parameter.
"""

class HelpException(Exception):
    pass

kyber_instances = {
    "kyber160" : {'n': 160, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber176" : {'n': 176, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber192" : {'n': 192, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber208" : {'n': 208, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber224" : {'n': 224, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber240" : {'n': 240, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber256" : {'n': 256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber512" : {'n': 2*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(3), 'dist': CB3},
    "kyber768" : {'n': 3*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(2), 'dist': CB2},
    "kyber1024": {'n': 4*256, 'q': 3329, 'st_dev_e': st_dev_central_binomial(2), 'dist': CB2}
}

def parse_all():
    if "-h" in argv:
        print(help_msg)
        he = HelpException("HelpException")
        raise he
    else:
        if any( "-n=" in tmp for tmp in argv ):
            # raise NotImplementedError("Only kyber{160,176,192,208,224,240,256,512,768,1024} instances are currently supported.")
            # assert any( "-q=" in tmp for tmp in argv ), "If -n= flag is present, -q= flag is required."
            # assert any( "-kappa=" in tmp for tmp in argv ), "If -n= flag is present, -q= flag is required."

            n, q, kappa, st_dev_e, dist = 140, 3329, 3, st_dev_central_binomial(3), centeredBinomial(3).PDF
            brk = 0
            for s in argv[1:]: #TODO: std_dev and dist
                if "-n=" in s:
                    # print("n found")
                    n = int(s[3:]) #the notions of n in "kybern" and in the code differ by a factor of 2
                    brk += 1
                    continue

                if "-q=" in s:
                    # print("q found")
                    q = int(s[3:])
                    brk += 1
                    continue

                if "-eta=" in s:
                    eta = int(s[5:])
                    st_dev_e = st_dev_central_binomial(eta)
                    dist = centeredBinomial(eta).PDF
                    continue

                if "-kmax=" in s:
                    # print("kappa found")
                    kappa = int(s[6:])
                    brk += 1
                    continue

                if brk >= 4:
                    break

            assert brk>= 3, "-n, -q or -kappa flag is not provided."
        else:
            assert any( "kyber" in tmp.lower() for tmp in argv ), "If -n= flag is present, -q= flag is required."
            brk = 0
            for s in argv:
                if "kyber" in s.lower():
                    # print("kyber found")
                    try:
                        KyberParam = kyber_instances[s.lower()]
                    except KeyError:
                        raise Exception(f"Kyber instance {s.lower()} not supported.")
                    n, q = KyberParam['n'], KyberParam['q']
                    st_dev_e, dist = KyberParam['st_dev_e'], KyberParam['dist']
                    brk += 1
                    continue

                if "-kmax=" in s:
                    # print("kappa found")
                    kappa = int(s[6:])
                    brk += 1
                    continue
                if brk >= 2:
                    break

            assert brk>=2, "kyber or -kmax flag is not provided."
        print(f"n, q, kappa, st_dev_e, dist: {n, q, kappa, st_dev_e, dist}")
        return n, q, kappa, st_dev_e, dist

if __name__=="__main__":
    n, q, kappa, st_dev_e, dist = parse_all()
    print( n, q, kappa, st_dev_e, dist )
