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
                        "H", "G", "I", "P",
                        "E", "B", 
                        "C", "T", "S" ]         # 8-state SS ( ! recent DSSP revision includes a 9th state)

# Create a hashing table for ss8Categories from Transformers.Basic.Vocabulary
const ss8Vocab = Vocabulary(ss8Categories, "-")
