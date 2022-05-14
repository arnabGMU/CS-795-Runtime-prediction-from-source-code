import os
import sys
import numpy as np
import random
import re
import pandas as pd
from typing import Dict, Optional
from subprocess import CalledProcessError, check_output
from pathlib import Path
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors

# export
def _isASCII(mthd: str) -> bool:
    """
    Check if the given method contains only ASCII characters. From https://stackoverflow.com/a/27084708/5768407.

    :param mthd: the method to verify contains only ASCII characters
    :returns: returns a boolean representing whether or not the given method contains only ASCII characters
    """
    try:
        mthd.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def remove_non_ascii(df: pd.DataFrame, n: Optional[int] = None) -> pd.DataFrame:
    """
    Remove all methods that contain non-ascii characters from a given pandas dataframe, not in-place.

    :param df: the pandas dataframe containing each method to be beautified
    :param n: the number of methods to evaluate. If none, the entire dataframe will be used
    :returns: returns a new dataframe without methods that contain non-ascii characters
    """
    if n is None:
        n = len(df)

    df = df.iloc[:n].copy()
    df = df[df.code.apply(_isASCII)]

    return df

# export
def _beautify(mthd: str) -> str:
    """
    Beautifies a given method using uncrustify with the sun.cfg style, i.e., Oracle's style.

    :param mthd: the method to beautify
    :returns: returns a beautified version of the given method
    """
    # get path of icodegen
    #icodegen_path = Path(icodegen.__path__[0])

    # create tmp file to store df contents for training tokenizer
    tmp_path = Path("./tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "tmp.java", "w") as f:
        f.write(mthd)

    try:
        beaut_mthd = check_output(
            [
                "./uncrustify",
                "-c",
                 "./sun.cfg",
                "-f",
                tmp_path / "tmp.java",
            ]
        ).decode("utf-8")
    except CalledProcessError as e:
        # Exception thrown when the method is malformed, i.e, it is missing a curly brace
        beaut_mthd = e.output.decode("utf-8")

    return beaut_mthd


def beautify_code(df: pd.DataFrame, n: Optional[int] = None) -> pd.DataFrame:
    """
    Beautify the methods in a pandas dataframe using uncrustify with the sun.cfg style, i.e., Oracle's style, not in-place.

    :param df: the pandas dataframe containing each method to be beautified
    :param n: the number of methods to evaluate. If none, the entire dataframe will be used
    :returns: returns a modified dataframe with the methods beautified
    """
    if n is None:
        n = len(df)

    df = df.iloc[:n].copy()
    df.code = df.code.apply(_beautify)

    return df

# export
# dicts of special tokens we are adding to the tokenizers so they do not get split

extra_tokens = {"<n>": "\n"}

# from https://docs.oracle.com/javase/tutorial/java/nutsandbolts/_keywords.html
java_reserved_tokens = {
    "<abstract>": "abstract",
    "<assert>": "assert",
    "<boolean>": "boolean",
    "<break>": "break",
    "<byte>": "byte",
    "<case>": "case",
    "<catch>": "catch",
    "<char>": "char",
    "<class>": "class",
    "<const>": "const",
    "<continue>": "continue",
    "<default>": "default",
    "<do>": "do",
    "<double>": "double",
    "<else>": "else",
    "<enum>": "enum",
    "<extends>": "extends",
    "<final>": "final",
    "<finally>": "finally",
    "<float>": "float",
    "<for>": "for",
    "<goto>": "goto",
    "<if>": "if",
    "<implements>": "implements",
    "<import>": "import",
    "<instanceof>": "instanceof",
    "<int>": "int",
    "<interface>": "interface",
    "<long>": "long",
    "<native>": "native",
    "<new>": "new",
    "<package>": "package",
    "<private>": "private",
    "<protected>": "protected",
    "<public>": "public",
    "<return>": "return",
    "<short>": "short",
    "<static>": "static",
    "<strictfp>": "strictfp",
    "<super>": "super",
    "<switch>": "switch",
    "<synchronized>": "synchronized",
    "<this>": "this",
    "<throw>": "throw",
    "<throws>": "throws",
    "<transient>": "transient",
    "<try>": "try",
    "<void>": "void",
    "<volatile>": "volatile",
    "<while>": "while",
}

# from https://docs.oracle.com/javase/tutorial/java/nutsandbolts/opsummary.html
java_operator_tokens = {
    "<=>": "=",
    "<+>": "+",
    "<->": "-",
    "<*>": "*",
    "</>": "/",
    "<%>": "%",
    "<++>": "++",
    "<-->": "--",
    "<!>": "!",
    "<==>": "==",
    "<!=>": "!=",
    "<greater>": ">",
    "<greater_equal>": ">=",
    "<lesser>": "<",
    "<lesser_equal>": "<=",
    "<&&>": "&&",
    "<||>": "||",
    "<?>": "?",
    "<:>": ":",
    "<~>": "~",
    "<double_lesser>": "<<",
    "<double_greater>": ">>",
    "<triple_greater>": ">>>",
    "<&>": "&",
    "<^>": "^",
    "<|>": "|",
}

java_structural_tokens = {
    "<{>": "{",
    "<}>": "}",
    "<[>": "[",
    "<]>": "]",
    "<lesser>": "<",
    "<greater>": ">",
    "<(>": "(",
    "<)>": ")",
    "<;>": ";",
}

java_extra_tokens = {
    "<@>": "@",
    "<...>": "...",
    "<null>": "null",
    "<true>": "true",
    "<false>": "false",
}

# combination of all dictionaries
java_special_tokens = {
    **java_reserved_tokens,
    **java_operator_tokens,
    **java_structural_tokens,
    **java_extra_tokens,
    **extra_tokens,
}

# export
def _replace_toks(mthd: str, spec_toks: Dict[str, str]) -> str:
    """
    Helper function for replacing all special tokens in a given method. This will replace longer special tokens first in order to not mistakenly breakup a special token that is part of a longer sequence. Adapted from https://stackoverflow.com/a/6117124/5768407 and https://stackoverflow.com/a/11753945/5768407

    :param mthd: the method to have its special tokens replaced
    :param spec_toks: a dictionary containing the special tokens to replace and the new tokens to replace them with
    :returns: returns the method with its special tokens replaced
    """
    # construct escaped versions of keys for running through regex
    spec_toks = dict(
        (re.escape(v), k)
        for k, v in sorted(
            java_special_tokens.items(), key=lambda x: len(x[1]), reverse=True
        )
    )
    # construct regex pattern for finding all special tokens in a method
    pattern = re.compile("|".join(spec_toks.keys()))
    # replace all special tokens in a method
    mthd = pattern.sub(lambda m: spec_toks[re.escape(m.group(0))], mthd)

    return mthd


def replace_special_tokens(
    df: pd.DataFrame, spec_toks: Dict[str, str], n: Optional[int] = None
) -> pd.DataFrame:
    """
    Replace all the special tokens in a pandas dataframe.

    :param df: the pandas dataframe containing each method to replace special tokens in
    :param n: the number of methods to evaluate. If none, the entire dataframe will be used
    :returns: returns a modified dataframe with the special tokens replaced
    """
    if n is None:
        n = len(df)

    df = df.iloc[:n].copy()
    df.code = df.code.apply(lambda mthd: _replace_toks(mthd, spec_toks))

    return df

# export
def train_tokenizer(
    df: pd.DataFrame,
    spec_toks: Dict[str, str],
    max_length: int,
    n: Optional[int] = None,
    vocab_sz: Optional[int] = 10000,
    min_freq: Optional[int] = 2,
    output: Optional[Path] = None,
) -> Tokenizer:
    """
    Train a ByteLevel BPE tokenizer on a given pandas dataframe. Code adapted from https://github.com/huggingface/tokenizers/tree/master/bindings/python.

    :param df: the pandas dataframe containing each method to have the tokenizer train on
    :param spec_toks: dict of special tokens to add to the tokenizers so they do not get split
    :param n: the number of methods to evaluate. If none, the entire dataframe will be used
    :param vocab_sz: the maximum vocabulary size of the trained tokenizer. Defaulted was selected from: Big Code != Big Vocabulary: Open-Vocabulary Models for Source Code
    :param min_freq: the minimum frequency a token has to occur to be considered
    :returns: returns a trained ByteLevel BPE tokenizer
    """
    if n is None:
        n = len(df)

    # create tmp file to store df contents for training tokenizer
    tmp_path = Path("./tmp")
    tmp_path.mkdir(parents=True, exist_ok=True)
    with open(tmp_path / "tmp_tokenize.txt", "w") as f:
        f.write("\n".join(df.code.values[:n]))

    # initialize a tokenizer
    tokenizer = Tokenizer(models.BPE())

    # customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)

    # train tokenizer with data in tmp file
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_sz,
        min_frequency=min_freq,
        special_tokens=["<pad>", "<cls>", "<eos>"] + list(spec_toks.keys()),
    )
    tokenizer.train([str(tmp_path / "tmp_tokenize.txt")], trainer=trainer)
    tokenizer.enable_padding(max_length=max_length, pad_token="<pad>")
    tokenizer.enable_truncation(max_length)

    # save tokenizer if output path given
    if output is not None:
        tokenizer.save(output, pretty=True)

    return tokenizer

def get_tokenizer():
    dataset_file = open('./correct_dataset.txt', 'r')
    length = []
    tokenized_codes = []
    non_ascii_files = []
    filename_wrong = []
    for line in dataset_file.readlines()[1:]:
        filename = line.split(',')[0]
        try:
            java_file = open(filename,'r').read()
        except:
            filename_wrong.append(filename)
        df = pd.DataFrame([java_file], columns=["code"])
        df = beautify_code(df)
        df = remove_non_ascii(df)
        df = replace_special_tokens(df, java_special_tokens)
        try:
            tokenized_codes.append([df.code.values[0]])
            length.append(len(df.code.values[0].split()))
        except:
            non_ascii_files.append(filename)
        
    df = pd.DataFrame(tokenized_codes,columns=["code"])
    tokenizer = train_tokenizer(df, java_special_tokens, max_length=2000)
    return tokenizer
    
    
    