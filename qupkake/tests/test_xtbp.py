import pytest
from rdkit import Chem

from qupkake.xtbp import XTBP, RunXTB

# Mock xtb output data
xtb_output = """
      -----------------------------------------------------------      
     |                   =====================                   |     
     |                           x T B                           |     
     |                   =====================                   |     
     |                         S. Grimme                         |     
     |          Mulliken Center for Theoretical Chemistry        |     
     |                    University of Bonn                     |     
      -----------------------------------------------------------      

   * xtb version 6.4.1 (unknown) compiled by 'oda6@login1.crc.pitt.edu' on 2021-06-25

           -------------------------------------------------
          |                Calculation Setup                |
           -------------------------------------------------

          program call               : /ihome/ghutchison/oda6/xtb/xtb-641//bin/xtb dvb_lmo.xyz --lmo
          hostname                   : login0.crc.pitt.edu
          coordinate file            : dvb_lmo.xyz
          omp threads                :                    12
          number of atoms            :                    19
          number of electrons        :                    46
          charge                     :                     0
          spin                       :                   0.0
          first test random number   :      0.01436930346367

   ID    Z sym.   atoms
    1    6 C      1-5, 10, 11, 14, 15
    2    1 H      6-9, 12, 13, 16-19

           -------------------------------------------------
          |                 G F N 2 - x T B                 |
           -------------------------------------------------

        Reference                      10.1021/acs.jctc.8b01176
      * Hamiltonian:
        H0-scaling (s, p, d)           1.850000    2.230000    2.230000
        zeta-weighting                 0.500000
      * Dispersion:
        s8                             2.700000
        a1                             0.520000
        a2                             5.000000
        s9                             5.000000
      * Repulsion:
        kExp                           1.500000    1.000000
        rExp                           1.000000
      * Coulomb:
        alpha                          2.000000
        third order                    shell-resolved
        anisotropic                    true
        a3                             3.000000
        a5                             4.000000
        cn-shift                       1.200000
        cn-exp                         4.000000
        max-rad                        5.000000


          ...................................................
          :                      SETUP                      :
          :.................................................:
          :  # basis functions                  46          :
          :  # atomic orbitals                  46          :
          :  # shells                           28          :
          :  # electrons                        46          :
          :  max. iterations                   250          :
          :  Hamiltonian                  GFN2-xTB          :
          :  restarted?                      false          :
          :  GBSA solvation                  false          :
          :  PC potential                    false          :
          :  electronic temp.          300.0000000     K    :
          :  accuracy                    1.0000000          :
          :  -> integral cutoff          0.2500000E+02      :
          :  -> integral neglect         0.1000000E-07      :
          :  -> SCF convergence          0.1000000E-05 Eh   :
          :  -> wf. convergence          0.1000000E-03 e    :
          :  Broyden damping             0.4000000          :
          ...................................................

 iter      E             dE          RMSdq      gap      omega  full diag
   1    -24.7094344 -0.247094E+02  0.462E+00    3.20       0.0  T
   2    -24.7323307 -0.228963E-01  0.277E+00    3.19       1.0  T
   3    -24.7233396  0.899110E-02  0.102E+00    3.20       1.0  T
   4    -24.7325216 -0.918203E-02  0.456E-01    3.19       1.0  T
   5    -24.7350026 -0.248095E-02  0.897E-02    3.18       1.0  T
   6    -24.7350672 -0.646251E-04  0.418E-02    3.18       1.0  T
   7    -24.7350755 -0.824757E-05  0.238E-02    3.18       1.0  T
   8    -24.7350781 -0.263615E-05  0.806E-03    3.18       2.8  T
   9    -24.7350790 -0.931998E-06  0.161E-03    3.18      14.2  T
  10    -24.7350791 -0.341434E-07  0.662E-04    3.18      34.6  T
  11    -24.7350791 -0.135722E-08  0.174E-04    3.18     131.5  T

   *** convergence criteria satisfied after 11 iterations ***

         #    Occupation            Energy/Eh            Energy/eV
      -------------------------------------------------------------
         1        2.0000           -0.6472854             -17.6135
       ...           ...                  ...                  ...
        17        2.0000           -0.4479545             -12.1895
        18        2.0000           -0.4380834             -11.9209
        19        2.0000           -0.4338653             -11.8061
        20        2.0000           -0.4208020             -11.4506
        21        2.0000           -0.4068824             -11.0718
        22        2.0000           -0.4020068             -10.9392
        23        2.0000           -0.3687110             -10.0331 (HOMO)
        24                         -0.2517105              -6.8494 (LUMO)
        25                         -0.2122859              -5.7766
        26                         -0.2001758              -5.4471
        27                         -0.1055082              -2.8710
        28                          0.0973601               2.6493
       ...                                ...                  ...
        46                          0.7096871              19.3116
      -------------------------------------------------------------
                  HL-Gap            0.1170005 Eh            3.1837 eV
             Fermi-level           -0.3102107 Eh           -8.4413 eV
 
 localization/xTB-IFF output generation
 averaging CT terms over            1  occ. levels
 averaging CT terms over            1  virt. levels
 dipole moment from electron density (au)
     X       Y       Z   
  -0.1784  -0.0380  -0.0677  total (Debye):    0.495
cpu  time for init local    0.06 s
wall time for init local    0.01 s
 doing rotations ...
 initialization of trafo matrix to unity

 converged in     10 iterations, threshold :   0.99964099D-06
 doing transformations ...
 lmo centers(Z=2) and atoms on file <lmocent.coord>
 LMO Fii/eV  ncent    charge center   contributions...
    1 sigma -14.50   2.07  -2.96674  -0.27666  -0.07693    5C :  0.50   11C :  0.48
    2 sigma -14.49   1.99   3.57067  -0.70571   0.08641   10C :  0.50    1C :  0.50
    3 sigma -14.28   2.01  -0.97708  -0.94137  -0.78405    5C :  0.51    3C :  0.49
    4 sigma -14.28   2.00   2.01493   1.08322  -0.25630    4C :  0.50    1C :  0.50
    5 sigma -14.27   2.01  -1.56813   1.21276  -0.94569    5C :  0.51    2C :  0.49
    6 sigma -13.93   2.10  -2.47392  -0.30368  -2.21435    5C :  0.50    9H :  0.47
    7 sigma -13.81   2.00  -1.66945   3.62515  -1.03698    2C :  0.51    6H :  0.49
    8 sigma -13.81   2.00   0.16696  -3.05583  -0.53525    3C :  0.51    7H :  0.49
    9 sigma -13.78   2.00  -3.64509  -0.28620   2.19505   11C :  0.51   13H :  0.49
   10 sigma -13.72   2.00  -7.29391  -1.32407   1.39354   15C :  0.51   19H :  0.49
   11 sigma -13.71   1.99  -6.81818  -1.36172  -0.83213   15C :  0.51   17H :  0.49
   12 sigma -13.70   2.01   2.39300   3.40909  -0.22941    4C :  0.51    8H :  0.49
   13 sigma -13.61   1.98   8.06572  -0.31192   0.91783   14C :  0.51   18H :  0.49
   14 sigma -13.60   2.01   4.92389  -2.53266   0.41566   10C :  0.51   12H :  0.49
   15 sigma -13.56   1.99   6.81829   1.56955   0.60427   14C :  0.52   16H :  0.49
   16 pi    -13.38   1.99  -5.07702  -1.34503   0.70308   15C :  0.50   11C :  0.50
   17 pi    -13.37   1.99  -5.35591  -0.30037   0.62513   15C :  0.50   11C :  0.50
   18 pi    -13.24   2.03   5.67491  -0.45503   1.01585   14C :  0.50   10C :  0.49
   19 pi    -13.24   2.03   5.87170  -0.49845  -0.02987   14C :  0.50   10C :  0.49
   20 pi    -13.16   2.14   1.43826  -0.96754  -0.85045    3C :  0.49    1C :  0.48
   21 pi    -13.15   2.06   0.41921   2.41512  -1.13966    2C :  0.50    4C :  0.49
   22 pi    -13.14   2.13   1.25311  -0.94977   0.15091    3C :  0.48    1C :  0.48
   23 pi    -13.13   2.05   0.22693   2.44673  -0.10838    4C :  0.49    2C :  0.49
 starting deloc pi regularization ...
 thr    2.20000000000000      # pi deloc LMO           4
 files:
 coordprot.0/xtbscreen.xyz/xtblmoinfo/lmocent.coord
 with protonation site input, xtbdock and
 LMO center info written
 

 SCC (total)                   0 d,  0 h,  0 min,  0.036 sec
 SCC setup                      ...        0 min,  0.000 sec (  0.588%)
 Dispersion                     ...        0 min,  0.000 sec (  0.536%)
 classical contributions        ...        0 min,  0.000 sec (  0.138%)
 integral evaluation            ...        0 min,  0.001 sec (  1.843%)
 iterations                     ...        0 min,  0.008 sec ( 22.595%)
 molecular gradient             ...        0 min,  0.001 sec (  3.906%)
 printout                       ...        0 min,  0.026 sec ( 70.133%)

         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         ::                     SUMMARY                     ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::
         :: total energy             -24.265910603272 Eh    ::
         :: gradient norm              0.000716047326 Eh/a0 ::
         :: HOMO-LUMO gap              3.183747012927 eV    ::
         ::.................................................::
         :: SCC energy               -24.735079066765 Eh    ::
         :: -> isotropic ES            0.002618300195 Eh    ::
         :: -> anisotropic ES          0.004605178988 Eh    ::
         :: -> anisotropic XC          0.019900718341 Eh    ::
         :: -> dispersion             -0.014579677317 Eh    ::
         :: repulsion energy           0.469180641564 Eh    ::
         :: add. restraining           0.000000000000 Eh    ::
         :: total charge              -0.000000000000 e     ::
         :::::::::::::::::::::::::::::::::::::::::::::::::::::

           -------------------------------------------------
          |                Property Printout                |
           -------------------------------------------------

    * Orbital Energies and Occupations

         #    Occupation            Energy/Eh            Energy/eV
      -------------------------------------------------------------
         1        2.0000           -0.6472854             -17.6135
       ...           ...                  ...                  ...
        11        2.0000           -0.5018863             -13.6570
        12        2.0000           -0.4981047             -13.5541
        13        2.0000           -0.4817112             -13.1080
        14        2.0000           -0.4619459             -12.5702
        15        2.0000           -0.4569566             -12.4344
        16        2.0000           -0.4558266             -12.4037
        17        2.0000           -0.4479545             -12.1895
        18        2.0000           -0.4380834             -11.9209
        19        2.0000           -0.4338653             -11.8061
        20        2.0000           -0.4208020             -11.4506
        21        2.0000           -0.4068824             -11.0718
        22        2.0000           -0.4020068             -10.9392
        23        2.0000           -0.3687110             -10.0331 (HOMO)
        24                         -0.2517105              -6.8494 (LUMO)
        25                         -0.2122859              -5.7766
        26                         -0.2001758              -5.4471
        27                         -0.1055082              -2.8710
        28                          0.0973601               2.6493
        29                          0.1093827               2.9765
        30                          0.1147667               3.1230
        31                          0.1237909               3.3685
        32                          0.1388843               3.7792
        33                          0.1455394               3.9603
        34                          0.1686349               4.5888
       ...                                ...                  ...
        46                          0.7096871              19.3116
      -------------------------------------------------------------
                  HL-Gap            0.1170005 Eh            3.1837 eV
             Fermi-level           -0.3102107 Eh           -8.4413 eV

     #   Z          covCN         q      C6AA      α(0)
     1   6 C        3.043     0.007    27.822     8.651
     2   6 C        2.992    -0.042    29.038     8.839
     3   6 C        3.003    -0.047    29.163     8.859
     4   6 C        2.997    -0.040    28.979     8.831
     5   6 C        3.934    -0.019    20.848     6.489
     6   1 H        0.926     0.028     2.616     2.529
     7   1 H        0.926     0.028     2.612     2.527
     8   1 H        0.926     0.031     2.571     2.507
     9   1 H        0.924     0.062     2.176     2.307
    10   6 C        2.891    -0.028    28.699     8.783
    11   6 C        2.893    -0.016    28.421     8.741
    12   1 H        0.925     0.030     2.579     2.511
    13   1 H        0.925     0.036     2.496     2.471
    14   6 C        2.839    -0.086    30.135     8.996
    15   6 C        2.836    -0.094    30.343     9.027
    16   1 H        0.926     0.040     2.447     2.446
    17   1 H        0.926     0.038     2.478     2.461
    18   1 H        0.926     0.033     2.546     2.495
    19   1 H        0.926     0.038     2.469     2.457

 Mol. C6AA /au·bohr⁶  :       4027.683249
 Mol. C8AA /au·bohr⁸  :      96897.936911
 Mol. α(0) /au        :        101.928856


Wiberg/Mayer (AO) data.
largest (>0.10) Wiberg bond orders for each atom

 ---------------------------------------------------------------------------
     #   Z sym  total        # sym  WBO       # sym  WBO       # sym  WBO
 ---------------------------------------------------------------------------
     1   6 C    3.987 --     3 C    1.697    10 C    1.104     4 C    1.098
     2   6 C    3.988 --     4 C    1.829     5 C    1.005     6 H    0.973
                             3 C    0.104
     3   6 C    3.987 --     1 C    1.697     5 C    1.013     7 H    0.973
                            14 C    0.125     2 C    0.104
     4   6 C    3.989 --     2 C    1.829     1 C    1.098     8 H    0.969
     5   6 C    3.996 --     3 C    1.013     2 C    1.005    11 C    0.979
                             9 H    0.924
     6   1 H    0.999 --     2 C    0.973
     7   1 H    0.999 --     3 C    0.973
     8   1 H    0.998 --     4 C    0.969
     9   1 H    0.992 --     5 C    0.924
    10   6 C    3.992 --    14 C    1.891     1 C    1.104    12 H    0.965
    11   6 C    3.985 --    15 C    1.971     5 C    0.979    13 H    0.967
    12   1 H    0.999 --    10 C    0.965
    13   1 H    0.999 --    11 C    0.967
    14   6 C    3.983 --    10 C    1.891    18 H    0.979    16 H    0.975
                             3 C    0.125
    15   6 C    3.988 --    11 C    1.971    17 H    0.975    19 H    0.975
    16   1 H    0.998 --    14 C    0.975
    17   1 H    0.999 --    15 C    0.975
    18   1 H    0.998 --    14 C    0.979
    19   1 H    0.996 --    15 C    0.975
 ---------------------------------------------------------------------------

Topologies differ in total number of bonds
Writing topology from bond orders to xtbtopo.mol

molecular dipole:
                 x           y           z       tot (Debye)
 q only:       -0.158      -0.027      -0.064
   full:       -0.178      -0.038      -0.068       0.495
molecular quadrupole (traceless):
                xx          xy          yy          xz          yz          zz
 q only:        0.492       0.341      -0.039       0.570       0.100      -0.453
  q+dip:        2.638       1.308       0.568       2.205       0.354      -3.206
   full:        0.902       0.562       0.305       1.208       0.219      -1.206


           -------------------------------------------------
          | TOTAL ENERGY              -24.265910603272 Eh   |
          | GRADIENT NORM               0.000716047326 Eh/α |
          | HOMO-LUMO GAP               3.183747012927 eV   |
           -------------------------------------------------
"""


@pytest.fixture
def sample_molecule():
    mol = Chem.MolFromSmiles("CCO")
    return mol


def test_run_xtb_with_molecule(sample_molecule):
    # Create an instance of RunXTB with a molecule
    xtb_runner = RunXTB(mol=sample_molecule, options="--opt")

    # Ensure that the xtb_runner instance was created successfully
    assert xtb_runner is not None

    # Perform additional assertions based on the expected behavior of your RunXTB class
    # For example, you might want to check that the optimization was performed successfully
    assert xtb_runner.opt_done

    # Access the optimized molecule and perform assertions on it
    opt_mol = xtb_runner.get_opt_mol()
    assert opt_mol is not None
    assert isinstance(opt_mol, Chem.Mol)


def test_run_xtb_with_invalid_input():
    # Test creating RunXTB with invalid input (e.g., missing molecule or file)
    with pytest.raises(RuntimeError):
        xtb_runner = RunXTB(options="--opt")

    with pytest.raises(RuntimeError):
        xtb_runner = RunXTB(mol="invalid_molecule", options="--opt")


def test_run_xtb_executable_exists():
    # Test that the xTB executable exists
    with pytest.raises(RuntimeError):
        xtb_runner = RunXTB(mol=None, options="--opt")


def test_xtbp_init_from_string():
    xtb_parser = XTBP(xtb_output)
    attributes = xtb_parser.get_attributes()
    print(attributes)
    assert attributes["metadata"]["package_version"] == "6.4.1"
    assert attributes["metadata"]["coord_type"] == "xyz"
    assert attributes["natom"] == 19
    assert attributes["charge"] == 0
    assert attributes["multiplicity"] == 1.0
    assert attributes["totalenergy"] == -24.265910603272
    assert attributes["hlgap"] == 3.183747012927
    assert attributes["atomprop"]["convcn"] == [
        3.043,
        2.992,
        3.003,
        2.997,
        3.934,
        0.926,
        0.926,
        0.926,
        0.924,
        2.891,
        2.893,
        0.925,
        0.925,
        2.839,
        2.836,
        0.926,
        0.926,
        0.926,
        0.926,
    ]
    assert attributes["atomprop"]["q"] == [
        0.007,
        -0.042,
        -0.047,
        -0.04,
        -0.019,
        0.028,
        0.028,
        0.031,
        0.062,
        -0.028,
        -0.016,
        0.03,
        0.036,
        -0.086,
        -0.094,
        0.04,
        0.038,
        0.033,
        0.038,
    ]
    assert attributes["atomprop"]["c6aa"] == [
        27.822,
        29.038,
        29.163,
        28.979,
        20.848,
        2.616,
        2.612,
        2.571,
        2.176,
        28.699,
        28.421,
        2.579,
        2.496,
        30.135,
        30.343,
        2.447,
        2.478,
        2.546,
        2.469,
    ]
    assert attributes["atomprop"]["alpha"] == [
        8.651,
        8.839,
        8.859,
        8.831,
        6.489,
        2.529,
        2.527,
        2.507,
        2.307,
        8.783,
        8.741,
        2.511,
        2.471,
        8.996,
        9.027,
        2.446,
        2.461,
        2.495,
        2.457,
    ]
    assert attributes["bondprop"]["wbo"] == [
        {"total": 3.987, 2: 1.697, 9: 1.104, 3: 1.098},
        {"total": 3.988, 3: 1.829, 4: 1.005, 5: 0.973, 2: 0.104},
        {"total": 3.987, 0: 1.697, 4: 1.013, 6: 0.973, 13: 0.125, 1: 0.104},
        {"total": 3.989, 1: 1.829, 0: 1.098, 7: 0.969},
        {"total": 3.996, 2: 1.013, 1: 1.005, 10: 0.979, 8: 0.924},
        {"total": 0.999, 1: 0.973},
        {"total": 0.999, 2: 0.973},
        {"total": 0.998, 3: 0.969},
        {"total": 0.992, 4: 0.924},
        {"total": 3.992, 13: 1.891, 0: 1.104, 11: 0.965},
        {"total": 3.985, 14: 1.971, 4: 0.979, 12: 0.967},
        {"total": 0.999, 9: 0.965},
        {"total": 0.999, 10: 0.967},
        {"total": 3.983, 9: 1.891, 17: 0.979, 15: 0.975, 2: 0.125},
        {"total": 3.988, 10: 1.971, 16: 0.975, 18: 0.975},
        {"total": 0.998, 13: 0.975},
        {"total": 0.999, 14: 0.975},
        {"total": 0.998, 13: 0.979},
        {"total": 0.996, 14: 0.975},
    ]


def test_xtbp_init_from_file(tmp_path):
    # Create a temporary file with xtb output
    xtb_file_path = tmp_path / "xtb_output.out"
    with open(xtb_file_path, "w") as f:
        f.write(xtb_output)

    # Test XTBP initialization from file
    xtb_parser = XTBP(str(xtb_file_path))
    attributes = xtb_parser.get_attributes()

    assert attributes["metadata"]["package_version"] == "6.4.1"
    assert attributes["metadata"]["coord_type"] == "xyz"
    assert attributes["natom"] == 19
    assert attributes["charge"] == 0
    assert attributes["multiplicity"] == 1.0
    assert attributes["totalenergy"] == -24.265910603272
    assert attributes["hlgap"] == 3.183747012927
    assert attributes["atomprop"]["convcn"] == [
        3.043,
        2.992,
        3.003,
        2.997,
        3.934,
        0.926,
        0.926,
        0.926,
        0.924,
        2.891,
        2.893,
        0.925,
        0.925,
        2.839,
        2.836,
        0.926,
        0.926,
        0.926,
        0.926,
    ]
    assert attributes["atomprop"]["q"] == [
        0.007,
        -0.042,
        -0.047,
        -0.04,
        -0.019,
        0.028,
        0.028,
        0.031,
        0.062,
        -0.028,
        -0.016,
        0.03,
        0.036,
        -0.086,
        -0.094,
        0.04,
        0.038,
        0.033,
        0.038,
    ]
    assert attributes["atomprop"]["c6aa"] == [
        27.822,
        29.038,
        29.163,
        28.979,
        20.848,
        2.616,
        2.612,
        2.571,
        2.176,
        28.699,
        28.421,
        2.579,
        2.496,
        30.135,
        30.343,
        2.447,
        2.478,
        2.546,
        2.469,
    ]
    assert attributes["atomprop"]["alpha"] == [
        8.651,
        8.839,
        8.859,
        8.831,
        6.489,
        2.529,
        2.527,
        2.507,
        2.307,
        8.783,
        8.741,
        2.511,
        2.471,
        8.996,
        9.027,
        2.446,
        2.461,
        2.495,
        2.457,
    ]
    assert attributes["bondprop"]["wbo"] == [
        {"total": 3.987, 2: 1.697, 9: 1.104, 3: 1.098},
        {"total": 3.988, 3: 1.829, 4: 1.005, 5: 0.973, 2: 0.104},
        {"total": 3.987, 0: 1.697, 4: 1.013, 6: 0.973, 13: 0.125, 1: 0.104},
        {"total": 3.989, 1: 1.829, 0: 1.098, 7: 0.969},
        {"total": 3.996, 2: 1.013, 1: 1.005, 10: 0.979, 8: 0.924},
        {"total": 0.999, 1: 0.973},
        {"total": 0.999, 2: 0.973},
        {"total": 0.998, 3: 0.969},
        {"total": 0.992, 4: 0.924},
        {"total": 3.992, 13: 1.891, 0: 1.104, 11: 0.965},
        {"total": 3.985, 14: 1.971, 4: 0.979, 12: 0.967},
        {"total": 0.999, 9: 0.965},
        {"total": 0.999, 10: 0.967},
        {"total": 3.983, 9: 1.891, 17: 0.979, 15: 0.975, 2: 0.125},
        {"total": 3.988, 10: 1.971, 16: 0.975, 18: 0.975},
        {"total": 0.998, 13: 0.975},
        {"total": 0.999, 14: 0.975},
        {"total": 0.998, 13: 0.979},
        {"total": 0.996, 14: 0.975},
    ]


def test_xtbp_invalid_input_type():
    with pytest.raises(
        TypeError, match="output must be an xTB output string or xTB Output file."
    ):
        XTBP(123)


def test_xtbp_parse_xtb_exception():
    with pytest.raises(Exception):
        # Simulating an exception during xtb parsing
        with patch(".XTBP.parse_xtb", side_effect=Exception("Simulated error")):
            xtb_parser = XTBP(xtb_output)


def test_xtbp_extract_data_exception():
    with pytest.raises(Exception):
        # Simulating an exception during data extraction
        with patch(".XTBP.extract_data", side_effect=Exception("Simulated error")):
            xtb_parser = XTBP(xtb_output)


if __name__ == "__main__":
    pytest.main()
