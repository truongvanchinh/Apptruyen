"""Sửa lỗi OCR tiếng Việt dựa trên cấu trúc âm tiết.

Một âm tiết = phụ âm đầu + vần (+ thanh). Từ nào OCR ra không phải âm tiết
tiếng Việt hợp lệ (vd "TRLYỀN", "TPÊN") sẽ được thử thay các ký tự hay bị
nhận nhầm, lấy phương án đầu tiên hợp lệ ("TRUYỀN", "TRÊN").
"""

from __future__ import annotations

import re
import unicodedata

TONE_MARKS = {"̀", "́", "̃", "̉", "̣"}

ONSETS = sorted(
    ["", "b", "c", "ch", "d", "đ", "g", "gh", "gi", "h", "k", "kh", "l", "m", "n", "ng",
     "ngh", "nh", "p", "ph", "qu", "r", "s", "t", "th", "tr", "v", "x"],
    key=len, reverse=True,
)

RHYMES = set("""
a ai ao au ay am an ang anh ac ach ap at
ăm ăn ăng ăc ăp ăt
âu ây âm ân âng âc âp ât
e eo em en eng ec ep et
ê êu êm ên ênh êch êp êt
i ia iu im in inh ich ip it
iêu iêm iên iêng iêc iêp iêt
o oa oai oao oay oam oan oang oanh oac oach oap oat oăm oăn oăng oăc oăt
oe oeo oen oet oi om on ong oc op ot ooc oong
ô ôi ôm ôn ông ôc ôp ôt
ơ ơi ơm ơn ơp ơt
u ua uâ uân uâng uât uây ui um un ung uc up ut
uê uêch uênh uơ uy uya uyên uyêt uych uynh uyt uyu
uôi uôm uôn uông uôc uôt
ư ưa ưi ưu ưm ưn ưng ưc ưt
ươi ươu ươm ươn ương ươc ươp ươt
y yêu yêm yên yêng yêt
""".split())

# Ký tự OCR hay nhầm -> các ứng viên đúng (chữ hoa; chữ thường xử lý tương tự)
CONFUSIONS = {
    "L": "UI", "P": "R", "H": "UN", "E": "R", "B": "R", "4": "A", "0": "O", "1": "I",
    "5": "S", "8": "B", "U": "L", "I": "L", "Q": "O", "J": "I", "F": "P", "Y": "U",
}


def _strip_tone(word: str) -> str:
    decomposed = unicodedata.normalize("NFD", word.lower())
    return unicodedata.normalize("NFC", "".join(c for c in decomposed if c not in TONE_MARKS))


def is_syllable(word: str) -> bool:
    w = _strip_tone(word)
    if not w.isalpha():
        return False
    for onset in ONSETS:
        if not w.startswith(onset):
            continue
        rest = w[len(onset):]
        if onset == "qu" and (not rest or rest.startswith("u")):
            continue
        if rest.startswith("y") and rest != "y" and onset not in ("", "qu"):  # "yên" chỉ đứng đầu, sau phụ âm là "iên"
            continue
        if rest in RHYMES or (onset == "gi" and rest == ""):
            return True
    return False


def _fix_word(word: str) -> str:
    if is_syllable(word):
        return word
    for i, ch in enumerate(word):
        options = CONFUSIONS.get(ch.upper(), "")
        for opt in options:
            opt = opt if ch.isupper() or ch.isdigit() else opt.lower()
            candidate = word[:i] + opt + word[i + 1:]
            if is_syllable(candidate):
                return candidate
    return word


def fix_vietnamese(text: str) -> str:
    """Chỉ sửa khi câu rõ ràng là tiếng Việt (có dấu)."""
    if not re.search(r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]",
                     text.lower()):
        return text
    return re.sub(r"[^\W_]+", lambda m: _fix_word(m.group(0)), text)
