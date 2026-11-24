// Assert web/js/tokens.js wordGroups() == build/common.py word_groups() on every dumped case.
import fs from 'node:fs';
import { wordGroups } from '../web/js/tokens.js';
const cases = JSON.parse(fs.readFileSync(new URL('./out/grouping_cases.json', import.meta.url), 'utf8'));
let bad = 0;
for (const c of cases) {
  const js = wordGroups(c.tokens).map(g => [g.text, g.tokens, g.punct]);
  if (JSON.stringify(js) !== JSON.stringify(c.groups)) { if (bad++ < 3) console.log('MISMATCH', c.tokens.join(' '), '\n js', JSON.stringify(js), '\n py', JSON.stringify(c.groups)); }
}
console.log(`${cases.length} cases, ${bad} mismatches`);
process.exit(bad ? 1 : 0);
