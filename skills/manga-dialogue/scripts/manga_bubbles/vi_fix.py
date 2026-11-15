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
    # Thay ký tự hay bị nhận nhầm; nếu có bảng tần suất thì lấy phương án phổ biến nhất
    table = _freq() or {}
    candidates = []
    for i, ch in enumerate(word):
        for opt in CONFUSIONS.get(ch.upper(), ""):
            opt = opt if ch.isupper() or ch.isdigit() else opt.lower()
            candidate = word[:i] + opt + word[i + 1:]
            if is_syllable(candidate):
                candidates.append(candidate)
    if candidates:
        return max(candidates, key=lambda c: table.get(c.lower(), 0.0))  # hoà thì lấy phương án đầu
    # Ký tự rác dính ở đầu/cuối chữ do nét viền bong bóng ("WTIẾNG" -> "TIẾNG", "NGỖIC" -> "NGỖI")
    if table and len(word) >= 3:
        for candidate in (word[1:], word[:-1]):
            if is_syllable(candidate) and (table.get(candidate.lower()) or table.get(_freq_fix(candidate).lower())):
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


# =========================
# SỬA DẤU THEO TẦN SUẤT TỪ
# =========================
# OCR hay đọc ra một âm tiết hợp lệ nhưng sai dấu ("RỖI" thay vì "RỒI"). Thử các biến thể
# dấu thanh / dấu mũ trên cùng chữ gốc; chỉ đổi khi biến thể phổ biến hơn áp đảo.

_SHAPES = {"a": ["", "\u0302", "\u0306"], "e": ["", "\u0302"], "o": ["", "\u0302", "\u031b"], "u": ["", "\u031b"]}
_TONES = ["", "\u0300", "\u0301", "\u0309", "\u0303", "\u0323"]
_VOWELS = "aeiouy"
CHANGE_PENALTY = 0.05   # mỗi dấu bị đổi làm giảm điểm 20 lần
MIN_GAIN = 20           # biến thể phải phổ biến hơn chữ gốc ít nhất chừng này lần (sau khi phạt)
_freq_table: dict | None | bool = None


def _freq() -> dict | None:
    """Bảng tần suất âm tiết tiếng Việt từ gói `wordfreq` (tuỳ chọn, chạy offline)."""
    global _freq_table
    if _freq_table is None:
        try:
            from wordfreq import get_frequency_dict

            _freq_table = get_frequency_dict("vi")
        except Exception:
            _freq_table = False
    return _freq_table or None


def _variants(core: str):
    """Sinh (biến thể, số dấu bị đổi) của một âm tiết viết thường."""
    from itertools import product

    chars = [_split_char(c) for c in core]
    vowels = [i for i, (b, _, _) in enumerate(chars) if b in _VOWELS]
    if not vowels or len(vowels) > 3:
        return
    old_tone_pos = next((i for i in vowels if chars[i][2]), None)
    old_tone = chars[old_tone_pos][2] if old_tone_pos is not None else ""
    tone_positions = [old_tone_pos] if old_tone_pos is not None else vowels
    for shapes in product(*[_SHAPES.get(chars[i][0], [chars[i][1]]) for i in vowels]):
        shape_changes = sum(sh != chars[i][1] for i, sh in zip(vowels, shapes))
        if shape_changes > 1:
            continue
        for tone in _TONES:
            for pos in (tone_positions if tone else tone_positions[:1]):
                new = list(chars)
                for i, sh in zip(vowels, shapes):
                    new[i] = (chars[i][0], sh, tone if (tone and i == pos) else "")
                changes = shape_changes + (tone != old_tone)
                if changes == 1:   # chỉ đổi một dấu: đổi nhiều hơn là đoán liều
                    yield "".join(_join_char(*c) for c in new), changes


def _tone_of(word: str) -> str:
    return next((t for _, _, t in map(_split_char, word) if t), "")


def _freq_fix(word: str) -> str:
    table = _freq()
    if not table or not (word.isupper() or word.islower()) or len(word) > 7:
        return word
    core = word.lower()
    best, best_score = core, max(table.get(core, 0.0), 1e-7) * MIN_GAIN
    # chỉ ưu tiên huyền khi chữ mang dấu ngã không có trong bảng từ ("MÃY", "THẴNG")
    old_is_tilde = _tone_of(core) == TILDE and not table.get(core)
    for cand, changes in _variants(core):
        f = table.get(cand, 0.0)
        if f and is_syllable(cand):
            score = f * CHANGE_PENALTY ** changes
            if old_is_tilde and _tone_of(cand) == GRAVE:
                score *= 20  # huyền bị đọc thành ngã là lỗi OCR phổ biến nhất
            if score > best_score:
                best, best_score = cand, score
    return best.upper() if word.isupper() else best


# chữ dấu ngã -> các chữ đứng trước (đã bỏ dấu thanh) mà khi đó dấu ngã là đúng
TILDE_CONTEXTS = {
    "nãy": {"vưa", "luc", "khi", "ban", "hôi"},                                   # vừa nãy, lúc nãy...
    "não": {"bô", "đâu", "tri", "oc", "phiên", "sâu", "ao", "tây", "chât", "mang", "vo"},  # bộ não, phiền não...
}


def fix_vietnamese(text: str, protected: frozenset = frozenset()) -> str:
    """Chỉ sửa khi câu rõ ràng là tiếng Việt (có dấu).
    `protected`: các chữ đã được xác nhận (hai bộ OCR đọc giống nhau) - không sửa theo tần suất."""
    if not re.search(r"[ăâđêôơưàáạảãằắặẳẵầấậẩẫèéẹẻẽềếệểễìíịỉĩòóọỏõồốộổỗờớợởỡùúụủũừứựửữỳýỵỷỹ]",
                     text.lower()):
        return text

    tokens = re.findall(r"[^\W_]+", text)
    keep = set(protected)
    for prev, cur in zip(tokens, tokens[1:]):
        if prev.upper() == cur.upper():
            keep.add(cur.upper())          # từ láy lặp đôi, tiếng động ("KHỤ KHỤ")

    previous = [""]

    def fix(match: re.Match) -> str:
        word = match.group(0)
        before, previous[0] = previous[0], word
        # Chữ dấu ngã chỉ đúng trong vài ngữ cảnh ("vừa nãy", "bộ não"); còn lại là huyền bị đọc nhầm
        contexts = TILDE_CONTEXTS.get(word.lower())
        if contexts is not None and (word.isupper() or word.islower()):
            if _strip_tone(before) in contexts:
                return word
            return "".join(_join_char(b, sh, GRAVE if t == TILDE else t) for b, sh, t in map(_split_char, word))
        fixed = _tilde_fix(_fix_word(word))
        # chữ đã được xác nhận thì không đoán lại theo tần suất
        return fixed if word.upper() in keep else _freq_fix(fixed)

    return re.sub(r"[^\W_]+", fix, text)


def agreed_words(easy: str, tess: str) -> frozenset:
    """Các chữ mà cả hai bộ OCR đọc giống hệt nhau (kể cả dấu)."""
    words = lambda t: {w.strip(PUNCT).upper() for w in t.split()} - {""}
    return frozenset(words(easy) & words(tess))


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
