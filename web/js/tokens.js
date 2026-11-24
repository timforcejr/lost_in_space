// Group SentencePiece tokens into display words. MUST mirror build/common.py word_groups().
// Verification: tests/grouping_parity.mjs (4,022 sentences, 0 mismatches vs Python)

const SPECIAL = new Set(['<s>', '</s>', '<pad>', '<unk>', '<mask>']);
const GLUE = new Set(["'", '’', '-']);
const isPunct = p => !/[\p{L}\p{M}\p{N}]/u.test(p);
const isHan = p => /[一-鿿㐀-䶿]/.test(p);

/**
 * @param {string[]} tokens token strings for the whole sequence, special tokens included
 * @returns {{text:string, tokens:number[], punct:boolean, han:boolean}[]}
 * A new word starts at a SentencePiece boundary; punctuation and Han pieces are always their
 * own unit (Chinese has no spaces, so each piece the tokenizer produced is one unit).
 */
export function wordGroups(tokens) {
  const groups = [];
  let pending = false;
  tokens.forEach((t, i) => {
    if (SPECIAL.has(t)) return;
    const piece = t.replaceAll('▁', '');
    const starts = t.startsWith('▁');
    if (piece === '') { pending = true; return; }
    const punct = isPunct(piece), han = isHan(piece);
    const prev = groups.length ? groups[groups.length - 1] : null;
    const newWord = starts || pending;
    pending = false;
    if (prev && !newWord && !prev.punct && !prev.han) {
      if (GLUE.has(piece)) { prev.text += piece; prev.tokens.push(i); prev.glue = true; return; }
      if (!punct && !han) { prev.text += piece; prev.tokens.push(i); prev.glue = false; return; }
    }
    groups.push({ text: piece, tokens: [i], punct, han, glue: false });
  });
  return groups;
}

/** Best-effort language guess from the script, for labels only (never used by the model). null = cannot tell. */
export function guessLang(text) {
  if (/[ऀ-ॿ]/.test(text)) return 'hi';
  if (/[぀-ヿ가-힯]/.test(text)) return null;                        // Japanese kana or Korean: not one of the four
  if (/[一-鿿]/.test(text)) return 'zh';
  if (!/[A-Za-zÀ-ɏ]/.test(text) || /[\u0370-\u052F\u0590-\u08FF\u0E00-\u0E7F]/.test(text)) return null;   // no Latin letters, or Greek / Cyrillic / Hebrew / Arabic / Thai
  const spanish = (text.match(/[áéíóúñ¿¡]/gi) || []).length + (text.match(/\b(el|la|los|las|una|que|por|con|para|del|está)\b/gi) || []).length;
  return spanish >= 1 ? 'es' : 'en';
}
