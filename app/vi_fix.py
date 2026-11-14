"""Sửa lỗi OCR tiếng Việt dựa trên cấu trúc âm tiết.

Một âm tiết = phụ âm đầu + vần (+ thanh). Từ nào OCR ra không phải âm tiết
tiếng Việt hợp lệ (vd "TRLYỀN", "TPÊN") sẽ được thử thay các ký tự hay bị
nhận nhầm, lấy phương án đầu tiên hợp lệ ("TRUYỀN", "TRÊN").
"""

from __future__ import annotations

import re
import unicodedata

TONE_MARKS = {"\u0300", "\u0301", "\u0303", "\u0309", "\u0323"}  # huyền, sắc, ngã, hỏi, nặng
HORN, GRAVE, TILDE = "\u031b", "\u0300", "\u0303"
PUNCT = '.,!?…~:;"\'()-`/\\|_'

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


def _split_char(ch: str) -> tuple[str, str, str]:
    """Tách ký tự thành (chữ gốc, dấu mũ/móc, dấu thanh)."""
    d = unicodedata.normalize("NFD", ch)
    return (d[0], "".join(m for m in d[1:] if m not in TONE_MARKS),
            "".join(m for m in d[1:] if m in TONE_MARKS))


def _join_char(base: str, shape: str, tone: str) -> str:
    return unicodedata.normalize("NFC", base + shape + tone)


def _add_horns(word: str) -> str:
    chars = [_split_char(c) for c in word]
    for i in range(len(chars) - 1):
        (b1, s1, t1), (b2, s2, t2) = chars[i], chars[i + 1]
        if b1.lower() == "u" and b2.lower() == "o" and s1 in ("", HORN) and s2 in ("", HORN):
            chars[i], chars[i + 1] = (b1, HORN, t1), (b2, HORN, t2)
    return "".join(_join_char(*c) for c in chars)


def _fix_word(word: str) -> str:
    if is_syllable(word):
        return word
    # Thiếu dấu móc ở cả cụm: "ĐUỌC" -> "ĐƯỢC", "NGUÒI" -> "NGƯỜI", "TRƯÓC" -> "TRƯỚC"
    horned = _add_horns(word)
    if horned != word and is_syllable(horned):
        return horned
    for i, ch in enumerate(word):
        options = CONFUSIONS.get(ch.upper(), "")
        for opt in options:
            opt = opt if ch.isupper() or ch.isdigit() else opt.lower()
            candidate = word[:i] + opt + word[i + 1:]
            if is_syllable(candidate):
                return candidate
    return word


# EasyOCR hay đọc dấu huyền thành dấu ngã. Các âm tiết dưới đây ở dạng ngã gần như không
# xuất hiện trong lời thoại, còn dạng huyền thì rất phổ biến -> đổi về huyền.
TILDE_TO_GRAVE = set("""
mĩnh hĩnh thĩ cõn hỗi trỡi gĩ ngưỡi nhĩn tĩm lãm dãnh nhãng rỗi mã lã vãi đễu đão ngãi
bãn cãng lãnh thãnh hãnh vẫy nhiễu điễu nhưng trưỡng đưỡng thưỡng vĩ khĩ nẫy bẫy gẫn
phẫn tẫm đẫu cẫn lẫn thẫn dẫn chĩ kĩ sẽo ngỡi
""".split()) - {"nhưng"}


def _tilde_fix(word: str) -> str:
    if word.lower() not in TILDE_TO_GRAVE:
        return word
    return "".join(_join_char(b, s, GRAVE if t == TILDE else t) for b, s, t in map(_split_char, word))


def fix_vietnamese(text: str) -> str:
    """Chỉ sửa khi câu rõ ràng là tiếng Việt (có dấu)."""
    if not re.search(r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]",
                     text.lower()):
        return text
    return re.sub(r"[^\W_]+", lambda m: _tilde_fix(_fix_word(m.group(0))), text)


# =========================
# GHÉP 2 BỘ OCR
# =========================
# EasyOCR cho khung chữ sạch nhưng hay sai dấu có hệ thống (huyền -> ngã, mất móc Ư/Ơ, I -> L/U);
# Tesseract dính rác từ viền bong bóng nhưng đọc dấu chính xác hơn. Giữ khung của EasyOCR,
# chỉ mượn Tesseract để sửa đúng các kiểu nhầm đó.

def _key(word: str) -> str:
    w = "".join(_split_char(c)[0] for c in word.upper().strip(PUNCT))
    return w.replace("L", "I").replace("1", "I")


def _merge_word(easy: str, tess: str) -> str:
    core_e, core_t = easy.strip(PUNCT), tess.strip(PUNCT)
    if not core_e or len(core_e) != len(core_t):
        return easy
    out = []
    for ce, ct in zip(core_e, core_t):
        be, se, te = _split_char(ce)
        bt, st, tt = _split_char(ct)
        if be.upper() == bt.upper():
            if not se and st == HORN:
                se = HORN                          # U -> Ư, O -> Ơ
            if (te == TILDE and tt == GRAVE) or (not te and tt and se == st):
                te = tt                            # ngã -> huyền; thiếu dấu thanh
        elif be.upper() in "L1U" and bt.upper() == "I":
            be = "I" if be.isupper() or be.isdigit() else "i"
            se, te = st, tt
        out.append(_join_char(be, se, te))
    merged = "".join(out)
    if merged != core_e and (is_syllable(merged) or not is_syllable(core_e)):
        return easy.replace(core_e, merged, 1)
    return easy


def merge_ocr(easy: str, tess: str) -> str:
    """Sửa dấu trong kết quả EasyOCR dựa vào kết quả Tesseract của cùng một ảnh."""
    import difflib

    ew, tw = easy.split(), tess.split()
    matcher = difflib.SequenceMatcher(None, [_key(w) for w in ew], [_key(w) for w in tw], autojunk=False)
    out = list(ew)
    for op, i1, i2, j1, j2 in matcher.get_opcodes():
        if op == "equal" or (op == "replace" and i2 - i1 == j2 - j1):
            for i, j in zip(range(i1, i2), range(j1, j2)):
                ke, kt = _key(ew[i]), _key(tw[j])
                if len(ke) == len(kt) and sum(a != b for a, b in zip(ke, kt)) <= 1:
                    out[i] = _merge_word(ew[i], tw[j])
    return " ".join(out)
