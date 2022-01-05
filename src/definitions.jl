#####
## Static definitions for use with TintiNet.jl
#

## 1. Sequence (FASTA) Dictionary

const fastaCategories = [ "-",                  # Padding/unknown
                          "A", "C", "D", "E",   
                          "F", "G", "H", "I",
                          "K", "L", "M", "N",
                          "P", "Q", "R", "S",
                          "T", "V", "W", "Y",   # 20 natural aminoacids
                          "X" ]                 # any aminoacid

# Create a hashing table for FastaCategories from Transformers.Basic.Vocabulary
const fastaVocab = Vocabulary(fastaCategories, "-")

## 2. Structure (SS3) Dictionary

const ss3Categories = [ "-",                    # Padding/ Unknown
                        "H", "E", "C" ]         # 3-state SS

# Create a hashing table for ss3Categories from Transformers.Basic.Vocabulary
const ss3Vocab = Vocabulary(ss3Categories, "-")

## 3. Structure (SS8) Dictionary

const ss8Categories = [ "-",                    # Padding/ Unknown
                        "H", "G", "I", "P",     # HELIX types
                        "E", "B",               # STRAND Types
                        "C", "T", "S" ]         # COIL types ## 8-state SS ( ! recent DSSP revision includes a 9th state)

# Create a hashing table for ss8Categories from Transformers.Basic.Vocabulary
const ss8Vocab = Vocabulary(ss8Categories, "-")

## 4. G-R-G ASA table
# RSA is the residue's ASA in the protein divided by the residue's ASA calculated when the residue is in an unfolded state,
# which is modeled as a Gly-R-Gly tripeptide.
const rsa_denominator_dictionary = Dict(
    "A" => 113,
    "R" => 241,
    "N" => 158,
    "D" => 151,
    "C" => 140,
    "Q" => 189,
    "E" => 183,
    "G" => 85,
    "H" => 194,
    "I" => 182,
    "L" => 180,
    "K" => 211,
    "M" => 204,
    "F" => 218,
    "P" => 143,
    "S" => 122,
    "T" => 146,
    "W" => 259,
    "Y" => 229,
    "V" => 160
)
