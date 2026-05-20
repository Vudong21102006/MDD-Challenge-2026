#!/usr/bin/env python
# coding: utf-8
# Evaluates mispronunciation detection F1 score.
# Inputs:
#   ground_truth.csv  — must contain columns: 'canonical', 'transcript'
#   results.csv       — must contain column:  'predict'
# The two files are paired row-by-row.
# Output: F1 score written to scores.txt in the output directory.
#
# Logic mirrors Align.py + ins_del_cor_sub_analysis.py exactly.
# dic[key] slot mapping (per the original comment "0:ref 1:human 2:ops --- 3:human 4:our 5:ops"):
#   ref_human alignment  → arr[0]=ref,       arr[1]=human,    arr[2]=op_rh
#   human_our alignment  → arr[3]=human,     arr[4]=our,      arr[5]=op_ho
#   ref_our   alignment  → arr[6]=ref,       arr[7]=our,      arr[8]=op_ro

import csv
import argparse


# ---------------------------------------------------------------------------
# Needleman-Wunsch aligner (identical to metric.py)
# ---------------------------------------------------------------------------

def _align(seq1, seq2):
    GAP = -1; MATCH = 1; MISMATCH = -1

    n, m = len(seq1), len(seq2)
    score = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        score[i][0] = GAP * i
    for j in range(n + 1):
        score[0][j] = GAP * j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[j-1] == seq2[i-1]:
                s = MATCH
            elif seq1[j-1] == "<eps>" or seq2[i-1] == "<eps>":
                s = GAP
            else:
                s = MISMATCH
            score[i][j] = max(
                score[i-1][j-1] + s,
                score[i-1][j]   + GAP,
                score[i][j-1]   + GAP,
            )

    align1, align2 = [], []
    i, j = m, n
    while i > 0 and j > 0:
        if seq1[j-1] == seq2[i-1]:
            s = MATCH
        elif seq1[j-1] == "<eps>" or seq2[i-1] == "<eps>":
            s = GAP
        else:
            s = MISMATCH
        if score[i][j] == score[i-1][j-1] + s:
            align1.append(seq1[j-1]); align2.append(seq2[i-1])
            i -= 1; j -= 1
        elif score[i][j] == score[i][j-1] + GAP:
            align1.append(seq1[j-1]); align2.append("<eps>")
            j -= 1
        else:
            align1.append("<eps>"); align2.append(seq2[i-1])
            i -= 1
    while j > 0:
        align1.append(seq1[j-1]); align2.append("<eps>"); j -= 1
    while i > 0:
        align1.append("<eps>"); align2.append(seq2[i-1]); i -= 1

    align1.reverse(); align2.reverse()
    return align1, align2


def _ops(aligned1, aligned2):
    ops = []
    for r, h in zip(aligned1, aligned2):
        if   r != "<eps>" and h == "<eps>": ops.append("D")
        elif r == "<eps>" and h != "<eps>": ops.append("I")
        elif r != h:                        ops.append("S")
        else:                               ops.append("C")
    return ops


def _align_pair(s1, s2):
    """Strip '*' and '$', split, align, return (aligned1, aligned2, ops)."""
    seq1 = s1.replace("*", "").replace("$", "").split()
    seq2 = s2.replace("*", "").replace("$", "").split()
    a1, a2 = _align(seq1, seq2)
    return a1, a2, _ops(a1, a2)


# ---------------------------------------------------------------------------
# CSV reader
# ---------------------------------------------------------------------------

def _read_csv(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Main scoring function
# ---------------------------------------------------------------------------

def compute_f1(ground_truth_path, results_path):
    gt  = _read_csv(ground_truth_path)
    res = _read_csv(results_path)

    assert "canonical"  in gt[0],  "ground_truth.csv must have a 'canonical' column"
    assert "transcript" in gt[0],  "ground_truth.csv must have a 'transcript' column"
    assert "predict"    in res[0], "results.csv must have a 'predict' column"

    cor_cor = cor_nocor = 0
    sub_sub = sub_sub1  = sub_nosub = 0
    ins_ins = ins_ins1  = ins_noins = 0
    del_del = del_del1  = del_nodel = 0

    for gt_row, res_row in zip(gt, res):
        ref_str   = gt_row["canonical"]
        human_str = gt_row["transcript"]
        our_str   = res_row["predict"]

        # Three alignments — slot names match the original dic[key] layout:
        #   arr[0..2]: ref_human  (ref=arr[0], human=arr[1], op_rh=arr[2])
        #   arr[3..5]: human_our  (human=arr[3], our=arr[4], op_ho=arr[5])
        #   arr[6..8]: ref_our    (ref=arr[6],   our=arr[7], op_ro=arr[8])
        ref_seq,    human_seq,  op_rh = _align_pair(ref_str,   human_str)  # slots 0,1,2
        human_seq2, our_seq2,   op_ho = _align_pair(human_str, our_str)    # slots 3,4,5
        ref_seq3,   our_seq3,   op_ro = _align_pair(ref_str,   our_str)    # slots 6,7,8

        # ---- Deletion detection (arr[0,2,6,7,8]) -------------------------
        # Walk ref_seq (arr[0]) with op_rh (arr[2]) alongside
        # ref_seq3 (arr[6]), our_seq3 (arr[7]), op_ro (arr[8]).
        flag = 0
        for i in range(len(ref_seq)):
            if ref_seq[i] == "<eps>":
                continue
            while flag < len(ref_seq3) and ref_seq3[flag] == "<eps>":
                flag += 1
            if flag < len(ref_seq3) and ref_seq[i] == ref_seq3[flag]:
                if   op_rh[i] == "D" and op_ro[flag] == "D":
                    del_del  += 1
                elif op_rh[i] == "D" and op_ro[flag] != "D" and op_ro[flag] != "C":
                    del_del1 += 1
                elif op_rh[i] == "D" and op_ro[flag] != "D" and op_ro[flag] == "C":
                    del_nodel += 1
                flag += 1

        # ---- Correct / Sub / Ins detection (arr[0,1,2,3,4,5]) -----------
        # Walk human_seq (arr[1]) with op_rh (arr[2]) alongside
        # human_seq2 (arr[3]), our_seq2 (arr[4]), op_ho (arr[5]).
        # ref_seq (arr[0]) is co-indexed with human_seq via the same alignment.
        flag = 0
        for i in range(len(human_seq)):
            if human_seq[i] == "<eps>":
                continue
            while flag < len(human_seq2) and human_seq2[flag] == "<eps>":
                flag += 1
            if flag < len(human_seq2) and human_seq[i] == human_seq2[flag]:

                if   op_rh[i] == "C" and op_ho[flag] == "C":
                    cor_cor   += 1
                elif op_rh[i] == "C" and op_ho[flag] != "C":
                    cor_nocor += 1

                if   op_rh[i] == "S" and op_ho[flag] == "C":
                    sub_sub   += 1
                elif op_rh[i] == "S" and op_ho[flag] != "C" and ref_seq[i] != our_seq2[flag]:
                    sub_sub1  += 1
                elif op_rh[i] == "S" and op_ho[flag] != "C" and ref_seq[i] == our_seq2[flag]:
                    sub_nosub += 1

                if   op_rh[i] == "I" and op_ho[flag] == "C":
                    ins_ins   += 1
                elif op_rh[i] == "I" and op_ho[flag] != "C" and op_ho[flag] != "D":
                    ins_ins1  += 1
                elif op_rh[i] == "I" and op_ho[flag] != "C" and op_ho[flag] == "D":
                    ins_noins += 1

                flag += 1

    TR = sub_sub + sub_sub1 + del_del + del_del1 + ins_ins + ins_ins1
    FR = cor_nocor
    FA = sub_nosub + ins_noins + del_nodel

    precision = TR / (TR + FR) if (TR + FR) > 0 else 0.0
    recall    = TR / (TR + FA) if (TR + FA) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1

def compute_per(ground_truth_path, results_path):
    gt  = _read_csv(ground_truth_path)
    res = _read_csv(results_path)

    # We enforce the same strict checks
    assert "transcript" in gt[0],  "ground_truth.csv must have a 'transcript' column"
    assert "predict"    in res[0], "results.csv must have a 'predict' column"

    total_sub = 0
    total_del = 0
    total_ins = 0
    total_ref_len = 0  # N = S + D + C

    for gt_row, res_row in zip(gt, res):
        human_str = gt_row["transcript"]  # Your reference for PER
        our_str   = res_row["predict"]    # Your model's output

        # Using your existing 2-way alignment helper
        # Returns lists/strings of the aligned sequences and operation codes
        # op_ho contains 'C' (Correct), 'S' (Sub), 'D' (Del), 'I' (Ins)
        human_seq, our_seq, op_ho = _align_pair(human_str, our_str)

        # Count the operations directly from the alignment op-codes
        sub = op_ho.count("S")
        del_ = op_ho.count("D")
        ins = op_ho.count("I")
        cor = op_ho.count("C")

        total_sub += sub
        total_del += del_
        total_ins += ins
        
        # The reference length is the sum of Substitutions, Deletions, and Corrects.
        # (Insertions do not exist in the original reference).
        total_ref_len += (sub + del_ + cor)

    # Prevent division by zero if the reference files are completely empty
    if total_ref_len == 0:
        return 0.0

    # Calculate final PER
    per = (total_sub + total_del + total_ins) / total_ref_len
    return per


def compute_der(ground_truth_path, results_path):
    gt  = _read_csv(ground_truth_path)
    res = _read_csv(results_path)

    assert "canonical"  in gt[0]
    assert "transcript" in gt[0]
    assert "predict"    in res[0]

    sub_sub = sub_sub1  = sub_nosub = 0
    ins_ins = ins_ins1  = ins_noins = 0
    del_del = del_del1  = del_nodel = 0

    for gt_row, res_row in zip(gt, res):
        ref_str   = gt_row["canonical"]
        human_str = gt_row["transcript"]
        our_str   = res_row["predict"]

        ref_seq,    human_seq,  op_rh = _align_pair(ref_str,   human_str)
        human_seq2, our_seq2,   op_ho = _align_pair(human_str, our_str)
        ref_seq3,   our_seq3,   op_ro = _align_pair(ref_str,   our_str)

        # ---- Deletion detection -------------------------
        flag = 0
        for i in range(len(ref_seq)):
            if ref_seq[i] == "<eps>":
                continue
            while flag < len(ref_seq3) and ref_seq3[flag] == "<eps>":
                flag += 1
            if flag < len(ref_seq3) and ref_seq[i] == ref_seq3[flag]:
                if   op_rh[i] == "D" and op_ro[flag] == "D":
                    del_del  += 1
                elif op_rh[i] == "D" and op_ro[flag] != "D" and op_ro[flag] != "C":
                    del_del1 += 1
                elif op_rh[i] == "D" and op_ro[flag] != "D" and op_ro[flag] == "C":
                    del_nodel += 1
                flag += 1

        # ---- Sub / Ins detection ------------------------
        flag = 0
        for i in range(len(human_seq)):
            if human_seq[i] == "<eps>":
                continue
            while flag < len(human_seq2) and human_seq2[flag] == "<eps>":
                flag += 1
            if flag < len(human_seq2) and human_seq[i] == human_seq2[flag]:
                
                if   op_rh[i] == "S" and op_ho[flag] == "C":
                    sub_sub   += 1
                elif op_rh[i] == "S" and op_ho[flag] != "C" and ref_seq[i] != our_seq2[flag]:
                    sub_sub1  += 1
                elif op_rh[i] == "S" and op_ho[flag] != "C" and ref_seq[i] == our_seq2[flag]:
                    sub_nosub += 1

                if   op_rh[i] == "I" and op_ho[flag] == "C":
                    ins_ins   += 1
                elif op_rh[i] == "I" and op_ho[flag] != "C" and op_ho[flag] != "D":
                    ins_ins1  += 1
                elif op_rh[i] == "I" and op_ho[flag] != "C" and op_ho[flag] == "D":
                    ins_noins += 1

                flag += 1

    # TR: True Rejection (System correctly detected an error)
    TR = sub_sub + sub_sub1 + del_del + del_del1 + ins_ins + ins_ins1
    
    # FA: False Acceptance (System missed an error)
    FA = sub_nosub + ins_noins + del_nodel
    
    # DE: Diagnosis Error (System detected the error, but gave the wrong phoneme)
    DE = sub_sub1 + del_del1 + ins_ins1

    total_actual_errors = TR + FA

    der = DE / total_actual_errors if total_actual_errors > 0 else 0.0
    return der


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute MDD F1 from two CSV files.")
    parser.add_argument("ground_truth_path", help="Path to ground_truth.csv")
    parser.add_argument("results_path", help="Path to results.csv")
    parser.add_argument(
        "-o",
        "--output",
        dest="output_path",
        help="Optional path to write the score. If omitted, the score is printed.",
    )
    args = parser.parse_args()

    f1 = compute_f1(args.ground_truth_path, args.results_path)
    per = compute_per(args.ground_truth_path, args.results_path)
    der = compute_der(args.ground_truth_path, args.results_path)


    print("F1: {:f}".format(f1))
    print(f"PER: {per}")
    print(f"DER: {der}")