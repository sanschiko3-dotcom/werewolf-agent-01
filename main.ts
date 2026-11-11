// werewolf-agent-01 entry: a deterministic, scripted Werewolf player.
//
// No model calls at all: every decision is a pure function of the game's view
// and MAKEFUN_TRIAL_SEED, so the live pass and the replay are identical by
// construction, and a trial costs $0 of inference. The agent plays all four
// roles with explicit strategy:
//
// - Village side (seer / doctor / villager): Bayesian-ish suspicion scoring
//   over the public record (votes, exiles, deaths, role claims and
//   accusations parsed out of the debate), ground-truth seer results when
//   playing the seer, and coordinated vote/speech play to exile the most
//   suspicious seat. Day 1 with no information it abstains rather than exile
//   a random villager; from day 2 it votes its top suspect.
// - Werewolf: coordinates the night kill with its partner (the second wolf
//   always agrees with the first's pick so a kill lands), targets seer
//   claimants and accurate accusers first, steers day votes onto the villager
//   the village already suspects most, and defends its partner lightly when
//   the partner is accused.
//
// Determinism rules obeyed: no wall clock, no Math.random, no network beyond
// the env (no model gateway use at all). Every tie-break and every "pick"
// comes from a SHA-256 hash of (trial seed, game, day, log length, context).
// Decisions are answered one at a time, in the order /observe lists them.
// Node 22 standard library only. Run by the grader as `node /bot/main.ts`.

import { createHash } from 'node:crypto';

export const CONCURRENCY = 1; // decisions answered one at a time, in order

type Seat = number;
type Role = 'werewolf' | 'seer' | 'doctor' | 'villager';
type DecisionKind = 'speak' | 'vote' | 'kill' | 'protect' | 'inspect';

interface Player {
  seat: Seat;
  name: string;
  alive: boolean;
  role: Role | null;
}

interface LogEntry {
  seq: number;
  day: number;
  kind: string;
  seat?: Seat;
  target?: Seat | null;
  text?: string;
  round?: number;
  died?: Seat | null;
  role?: Role | null;
  tally?: { seat: Seat; votes: number }[];
  order?: Seat[];
  winner?: string | null;
  [k: string]: unknown;
}

interface View {
  day: number;
  phase: string;
  you: {
    seat: Seat;
    name: string;
    role: Role;
    alive: boolean;
    partner: { seat: Seat; name: string; alive: boolean } | null;
  };
  players: Player[];
  seerResults: { night: number; seat: Seat; werewolf: boolean }[] | null;
  lastProtect: { night: number; seat: Seat | null } | null;
  wolfVotes: { night: number; seat: Seat; target: Seat | null }[] | null;
  log: LogEntry[];
  outcome: unknown;
}

interface Pending {
  game: number;
  decision: DecisionKind;
  options?: Seat[];
  max_chars?: number;
  round?: number;
  view: View;
}

interface Observed {
  games: number;
  finished: number;
  done: boolean;
  pending: Pending[];
}

const isRecord = (v: unknown): v is Record<string, unknown> =>
  v !== null && typeof v === 'object' && !Array.isArray(v);

/** Deterministic 32-bit hash of a string (FNV-1a). */
export function hash32(s: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < s.length; i++) {
    h ^= s.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** A stable pick in [0, n) from the trial seed and a context string. */
export function pickIndex(seed: string, context: string, n: number): number {
  if (n <= 0) return 0;
  const digest = createHash('sha256').update(`${seed}|${context}`).digest();
  return digest.readUInt32BE(0) % n;
}

/** A statement folded to one line the env's way. */
export function foldStatement(text: string): string {
  return text
    .replace(/[\t\n\v\f\r\u0085\u2028\u2029]+/g, ' ')
    .replace(/\p{Cc}/gu, '')
    .trim();
}

const who = (view: View, seat: Seat): string => {
  const p = view.players.find((x) => x.seat === seat);
  return p ? `seat ${seat} (${p.name})` : `seat ${seat}`;
};

const living = (view: View): Seat[] => view.players.filter((p) => p.alive).map((p) => p.seat);
const alive = (view: View, seat: Seat): boolean => view.players.some((p) => p.seat === seat && p.alive);
const revealedRole = (view: View, seat: Seat): Role | null => {
  const p = view.players.find((x) => x.seat === seat);
  return p ? p.role : null;
};

// ---------------------------------------------------------------------------
// Analysis: turn a view into suspicion scores and structured claims.
// ---------------------------------------------------------------------------

interface SeerClaim {
  seat: Seat;
  target: Seat;
  werewolf: boolean;
  day: number;
}

interface Accusation {
  from: Seat;
  target: Seat;
  day: number;
}

interface Analysis {
  /** Seats I know are werewolves (revealed, my seer results, or my partner's identity). */
  knownWolves: Set<Seat>;
  /** Seats I know are not werewolves (revealed, my seer clears, or myself when village). */
  knownClear: Set<Seat>;
  /** Suspicion per living seat, from public evidence only (votes, exiles, claims). */
  suspicion: Map<Seat, number>;
  /** Every seat that claimed to be the seer, with the results it claimed. */
  seerClaims: Map<Seat, SeerClaim[]>;
  /** Seer claimants with at least one claimed result matching a known fact. */
  corroboratedSeers: Set<Seat>;
  /** Every "X is a wolf" accusation found in speeches. */
  accusations: Accusation[];
  /** How many speeches each living seat has made (a proxy for table presence). */
  speechCount: Map<Seat, number>;
  /** Seats that voted for a revealed wolf at some point (village credit). */
  votedWolf: Set<Seat>;
  /** Seats that voted for a revealed villager at some point. */
  votedVillager: Set<Seat>;
  /** The seat I spoke about most recently this day (to avoid repeating myself). */
  myLastSpeechToday: string | null;
}

const SEAT_RE = /seat\s+(\d+)/gi;

/** Find seat numbers mentioned in a text, plus names of players. */
function mentionedSeats(text: string, view: View): Seat[] {
  const out = new Set<Seat>();
  for (const m of text.matchAll(SEAT_RE)) {
    const n = Number(m[1]);
    if (n >= 1 && n <= 8) out.add(n);
  }
  const low = text.toLowerCase();
  for (const p of view.players) {
    if (p.name.length >= 3 && low.includes(p.name.toLowerCase())) out.add(p.seat);
  }
  return [...out];
}

function claimsSeer(text: string): boolean {
  const t = text.toLowerCase();
  return (
    /\b(i am|i'm|im|i'm the|i am the)\b[^.]{0,25}\bseer\b/.test(t) ||
    /\bseer here\b/.test(t) ||
    /\bthe seer is me\b/.test(t) ||
    /\bi (checked|inspected)\b/.test(t)
  );
}

/** Parse "I checked/inspected seat N ... wolf / not a wolf" claims out of a speech. */
function parseInspectionClaims(text: string, from: Seat, day: number, view: View): SeerClaim[] {
  const out: SeerClaim[] = [];
  const low = text.toLowerCase();
  // Scan every mentioned seat for a nearby check + wolf/clear verdict.
  for (const target of mentionedSeats(text, view)) {
    if (target === from) continue;
    const idx = low.search(new RegExp(`seat\\s+${target}\\b`));
    const window = idx >= 0 ? low.slice(Math.max(0, idx - 60), idx + 80) : low;
    const saysWolf =
      /\b(wolf|werewolf)\b/.test(window) && !/\b(not|isn't|isnt|no)\b[^.]{0,12}\b(wolf|werewolf)\b/.test(window);
    const saysClear =
      /\b(not a wolf|not a werewolf|innocent|clear|villager|not the wolf)\b/.test(window);
    if (/\b(check|inspect|look|read|scry)\w*\b/.test(window) && (saysWolf || saysClear)) {
      out.push({ seat: from, target, werewolf: saysWolf && !saysClear, day });
    }
  }
  return out;
}

function parseAccusations(text: string, from: Seat, day: number, view: View): Accusation[] {
  const out: Accusation[] = [];
  const low = text.toLowerCase();
  for (const target of mentionedSeats(text, view)) {
    if (target === from) continue;
    const idx = low.search(new RegExp(`seat\\s+${target}\\b`));
    const window = idx >= 0 ? low.slice(Math.max(0, idx - 40), idx + 60) : low;
    if (
      /\b(wolf|werewolf)\b/.test(window) &&
      /\b(is|looks|sounds|seems|must be|has to be|think)\b/.test(window) &&
      !/\b(not|isn't|isnt)\b[^.]{0,12}\b(wolf|werewolf)\b/.test(window)
    ) {
      out.push({ from, target, day });
    }
  }
  return out;
}

export function analyze(view: View): Analysis {
  const me = view.you.seat;
  const knownWolves = new Set<Seat>();
  const knownClear = new Set<Seat>();
  for (const p of view.players) {
    if (p.role === 'werewolf') knownWolves.add(p.seat);
    else if (p.role !== null) knownClear.add(p.seat);
  }
  if (view.you.role === 'werewolf' && view.you.partner) knownWolves.add(view.you.partner.seat);
  if (view.you.role !== 'werewolf') knownClear.add(me);
  if (view.seerResults) {
    for (const r of view.seerResults) {
      if (r.werewolf) knownWolves.add(r.seat);
      else knownClear.add(r.seat);
    }
  }

  const suspicion = new Map<Seat, number>();
  for (const s of living(view)) if (!knownWolves.has(s) && !knownClear.has(s)) suspicion.set(s, 0);
  const bump = (seat: Seat, n: number) => {
    if (suspicion.has(seat)) suspicion.set(seat, (suspicion.get(seat) ?? 0) + n);
  };

  const seerClaims = new Map<Seat, SeerClaim[]>();
  const accusations: Accusation[] = [];
  const speechCount = new Map<Seat, number>();
  const votedWolf = new Set<Seat>();
  const votedVillager = new Set<Seat>();
  let myLastSpeechToday: string | null = null;

  for (const e of view.log) {
    if (e.kind === 'speech' && typeof e.text === 'string' && typeof e.seat === 'number') {
      speechCount.set(e.seat, (speechCount.get(e.seat) ?? 0) + 1);
      if (e.seat === me && e.day === view.day) myLastSpeechToday = e.text;
      if (claimsSeer(e.text)) {
        const claims = parseInspectionClaims(e.text, e.seat, e.day, view);
        if (claims.length > 0) {
          seerClaims.set(e.seat, [...(seerClaims.get(e.seat) ?? []), ...claims]);
        } else {
          seerClaims.set(e.seat, seerClaims.get(e.seat) ?? []);
        }
      }
      accusations.push(...parseAccusations(e.text, e.seat, e.day, view));
    }
    if (e.kind === 'vote' && typeof e.seat === 'number' && typeof e.target === 'number') {
      const role = revealedRole(view, e.target);
      if (role === 'werewolf') votedWolf.add(e.seat);
      else if (role !== null) votedVillager.add(e.seat);
    }
  }

  // Vote evidence: voting for a revealed wolf is village credit; voting for a
  // revealed villager is suspicious. Applied once per voter per outcome type.
  for (const s of votedWolf) bump(s, -2);
  for (const s of votedVillager) bump(s, 1);
  // Accusations: accusing a revealed wolf is credit; accusing a revealed
  // villager is suspicious.
  for (const a of accusations) {
    const role = revealedRole(view, a.target);
    if (role === 'werewolf') bump(a.from, -2);
    else if (role !== null) bump(a.from, 1);
  }
  // There is exactly one seer. If I am the seer, every other seer claimant
  // is a wolf — no exceptions. (If I am not the seer, the seer-claim
  // corroboration logic above is what separates the true seer from a faker.)
  if (view.you.role === 'seer') {
    for (const s of seerClaims.keys()) if (s !== me) knownWolves.add(s);
  }

  // Seer claims: a claim that contradicts a known fact is a lie (very wolfy);
  // one that matches a known fact corroborates the claimant. A claim naming a
  // wolf adds suspicion to the accused; a clear claim reduces it. An
  // uncorroborated claimant is itself mildly suspicious: wolves fake-claim
  // far more often than villagers do, and the true seer usually only claims
  // once they have a wolf to report.
  const corroboratedSeers = new Set<Seat>();
  for (const [claimer, claims] of seerClaims) {
    let corroborated = false;
    for (const c of claims) {
      const truth = knownWolves.has(c.target)
        ? true
        : knownClear.has(c.target)
          ? false
          : null;
      if (truth !== null && truth === c.werewolf) corroborated = true;
    }
    if (corroborated) corroboratedSeers.add(claimer);
  }
  for (const [claimer, claims] of seerClaims) {
    const corroborated = corroboratedSeers.has(claimer);
    for (const c of claims) {
      const truth = knownWolves.has(c.target)
        ? true
        : knownClear.has(c.target)
          ? false
          : null;
      if (truth !== null) {
        if (truth === c.werewolf) bump(claimer, -3);
        else bump(claimer, 6);
      }
      if (c.werewolf) bump(c.target, corroborated ? 2 : 1);
      else bump(c.target, -1);
    }
    if (!corroborated) bump(claimer, 1);
    // A day-1 claim is cheap for a wolf: the true seer rarely leads with a
    // claim before any death, while a wolf gains by framing a villager early.
    if (claims.some((c) => c.day === 1)) bump(claimer, 1);
  }
  // Two seer claimants cannot both be right when their claims clash on the
  // same seat (one says wolf, the other says clear), or when one accuses the
  // other of being a wolf: at least one of them is lying.
  const claimants = [...seerClaims.keys()];
  for (let i = 0; i < claimants.length; i++) {
    for (let j = i + 1; j < claimants.length; j++) {
      const a = claimants[i]!;
      const b = claimants[j]!;
      const ca = seerClaims.get(a) ?? [];
      const cb = seerClaims.get(b) ?? [];
      const clash =
        ca.some((x) => x.target === b && x.werewolf) ||
        cb.some((x) => x.target === a && x.werewolf) ||
        ca.some((x) => cb.some((y) => y.target === x.target && y.werewolf !== x.werewolf));
      if (clash) {
        bump(a, 2);
        bump(b, 2);
      }
    }
  }

  return { knownWolves, knownClear, suspicion, seerClaims, corroboratedSeers, accusations, speechCount, votedWolf, votedVillager, myLastSpeechToday };
}

/** Living seats sorted by suspicion desc, ties broken by seeded hash (deterministic). */
function rankedSuspects(view: View, a: Analysis, seed: string, game: number, exclude: Set<Seat> = new Set()): Seat[] {
  const cands = living(view).filter((s) => s !== view.you.seat && !exclude.has(s) && !a.knownClear.has(s));
  return cands.sort((x, y) => {
    const sx = a.suspicion.get(x) ?? (a.knownWolves.has(x) ? 100 : 0);
    const sy = a.suspicion.get(y) ?? (a.knownWolves.has(y) ? 100 : 0);
    if (sx !== sy) return sy - sx;
    const hx = hash32(`${seed}|${game}|${view.day}|${x}`);
    const hy = hash32(`${seed}|${game}|${view.day}|${y}`);
    return hx - hy;
  });
}

// ---------------------------------------------------------------------------
// Decisions.
// ---------------------------------------------------------------------------

function topSuspect(view: View, a: Analysis, seed: string, game: number): Seat | null {
  const ranked = rankedSuspects(view, a, seed, game);
  return ranked.length > 0 ? ranked[0]! : null;
}

function topScore(view: View, a: Analysis, seed: string, game: number): number {
  const t = topSuspect(view, a, seed, game);
  if (t === null) return -Infinity;
  if (a.knownWolves.has(t)) return 100;
  return a.suspicion.get(t) ?? 0;
}

/** The most credible seer claimant alive (claims matching known facts, else the only one). */
function credibleSeer(view: View, a: Analysis): Seat | null {
  const aliveClaimants = [...a.seerClaims.keys()].filter((s) => alive(view, s) && s !== view.you.seat);
  if (aliveClaimants.length === 0) return null;
  let best: Seat | null = null;
  let bestScore = -Infinity;
  for (const s of aliveClaimants) {
    const score = a.suspicion.get(s) ?? (a.knownWolves.has(s) ? 100 : 0);
    // Lower suspicion = more credible; a claimer I know is a wolf is not credible.
    if (a.knownWolves.has(s)) continue;
    const cred = -score;
    if (cred > bestScore) {
      bestScore = cred;
      best = s;
    }
  }
  return best;
}

function decideVote(view: View, a: Analysis, seed: string, game: number): Seat | null {
  const me = view.you.seat;
  if (!view.you.alive) return null;
  // A wolf I know about (my seer result, a revealed wolf, a trusted claim).
  const known = living(view).filter((s) => s !== me && a.knownWolves.has(s));
  if (known.length > 0) return known[0]!;
  // A seer claim is only worth following if it is corroborated by a known
  // fact, or (from day 2 on) the accused already looks suspicious on
  // independent evidence. An uncorroborated day-1 claim is cheap talk —
  // wolves fake-claim precisely to frame a villager, and following it cost
  // the village a game in testing.
  if (view.you.role !== 'werewolf') {
    const seer = credibleSeer(view, a);
    if (seer !== null) {
      const accused = (a.seerClaims.get(seer) ?? []).filter(
        (c) => c.werewolf && alive(view, c.target) && !a.knownClear.has(c.target),
      );
      if (accused.length > 0) {
        const t = accused[0]!.target;
        if (a.corroboratedSeers.has(seer)) return t;
        if (view.day >= 2 && (a.suspicion.get(t) ?? 0) >= 2) return t;
      }
    }
  }
  if (view.you.role === 'werewolf') {
    // Steer onto the villager the village already suspects most.
    const ranked = rankedSuspects(view, a, seed, game, new Set([view.you.partner?.seat ?? -1]));
    return ranked.length > 0 ? ranked[0]! : null;
  }
  // Village side: day 1 with no evidence, abstain rather than exile blind.
  if (view.day === 1 && topScore(view, a, seed, game) <= 0) return null;
  const t = topSuspect(view, a, seed, game);
  return t;
}

function decideKill(view: View, a: Analysis, seed: string, game: number): Seat | null {
  const me = view.you.seat;
  const partner = view.you.partner?.seat ?? null;
  const tonight = (view.wolfVotes ?? []).filter((v) => v.night === view.day);
  // Second wolf: agree with the partner's pick so the kill lands.
  const partnerPick = tonight.find((v) => v.seat === partner)?.target ?? null;
  if (partnerPick !== null && partnerPick !== undefined) return partnerPick;
  // First wolf (or partner abstained): pick the best target.
  const others = living(view).filter((s) => s !== me && s !== partner);
  if (others.length === 0) return null;
  // 1. A seer claimant: the most dangerous villager.
  const seer = credibleSeer(view, a);
  if (seer !== null && others.includes(seer)) return seer;
  // 2. Someone who accused me or my partner of being a wolf.
  const accusers = a.accusations
    .filter((x) => (x.target === me || x.target === partner) && others.includes(x.from))
    .map((x) => x.from);
  if (accusers.length > 0) {
    // The most recent accuser.
    return accusers[accusers.length - 1]!;
  }
  // 3. The most active villager the village trusts (low suspicion, many speeches).
  const byActivity = [...others].sort((x, y) => {
    const ax = a.speechCount.get(x) ?? 0;
    const ay = a.speechCount.get(y) ?? 0;
    if (ax !== ay) return ay - ax;
    const sx = a.suspicion.get(x) ?? 0;
    const sy = a.suspicion.get(y) ?? 0;
    if (sx !== sy) return sx - sy;
    return hash32(`${seed}|${game}|${x}`) - hash32(`${seed}|${game}|${y}`);
  });
  // Don't waste a kill on the villager the village is about to exile anyway.
  const villageTarget = topSuspect(view, a, seed, game);
  const pick = byActivity.find((s) => s !== villageTarget) ?? byActivity[0]!;
  return pick;
}

function decideProtect(view: View, a: Analysis, seed: string, game: number): Seat | null {
  const me = view.you.seat;
  const last = view.lastProtect?.seat ?? null;
  const cands = living(view).filter((s) => s !== me && s !== last);
  if (cands.length === 0) return null;
  // Protect a credible seer claimant first: they are the wolves' prime target.
  const seer = credibleSeer(view, a);
  if (seer !== null && cands.includes(seer)) return seer;
  // If nobody died last night and I protected someone, the wolves probably
  // went for them: they stay valuable, but I cannot repeat a protection.
  // Otherwise protect the most active trusted villager.
  const ranked = [...cands].sort((x, y) => {
    const ax = a.speechCount.get(x) ?? 0;
    const ay = a.speechCount.get(y) ?? 0;
    if (ax !== ay) return ay - ax;
    const sx = a.suspicion.get(x) ?? 0;
    const sy = a.suspicion.get(y) ?? 0;
    if (sx !== sy) return sx - sy;
    return hash32(`${seed}|${game}|${x}`) - hash32(`${seed}|${game}|${y}`);
  });
  return ranked[0]!;
}

function decideInspect(view: View, a: Analysis, seed: string, game: number): Seat | null {
  const me = view.you.seat;
  const done = new Set((view.seerResults ?? []).map((r) => r.seat));
  const cands = living(view).filter((s) => s !== me && !done.has(s) && !a.knownClear.has(s) && !a.knownWolves.has(s));
  if (cands.length === 0) return null;
  // Highest suspicion first; night 1 with no evidence, a seeded pick.
  const ranked = [...cands].sort((x, y) => {
    const sx = a.suspicion.get(x) ?? 0;
    const sy = a.suspicion.get(y) ?? 0;
    if (sx !== sy) return sy - sx;
    return hash32(`${seed}|${game}|${view.day}|${x}`) - hash32(`${seed}|${game}|${view.day}|${y}`);
  });
  return ranked[0]!;
}

// ---------------------------------------------------------------------------
// Speech.
// ---------------------------------------------------------------------------

function pickVariant(seed: string, game: number, day: number, round: number, variants: string[]): string {
  return variants[pickIndex(seed, `say|${game}|${day}|${round}`, variants.length)]!;
}

function decideSpeak(view: View, a: Analysis, seed: string, game: number, round: number): string | null {
  const me = view.you.seat;
  const role = view.you.role;
  if (!view.you.alive) return null;
  // Don't repeat the same case twice in one day: second round, pass unless
  // something new happened after my last speech (a seer claim, a death).
  if (round === 2 && a.myLastSpeechToday !== null) {
    const afterMe = view.log.some(
      (e) => e.kind === 'speech' && e.seat !== me && typeof e.text === 'string' && claimsSeer(e.text),
    );
    if (!afterMe) return null;
  }

  // Seer: if someone else claims to be the seer, they are lying — there is
  // exactly one seer, and it is me. Counter-claim at once and name them as
  // the wolf they are, citing my own latest result. This outranks a plain
  // reveal: an unanswered fake claim frames a villager and costs the game.
  if (role === 'seer') {
    const rivals = [...a.seerClaims.keys()].filter((s) => s !== me && alive(view, s));
    if (rivals.length > 0) {
      const liar = rivals[0]!;
      const results = (view.seerResults ?? []).slice().sort((x, y) => y.night - x.night);
      const latest = results[0];
      if (latest) {
        const verdict = latest.werewolf
          ? `they are a werewolf`
          : `not a wolf`;
        return `I'm the seer. I checked ${who(view, latest.seat)} last night and they are ${verdict}. ${who(view, liar)} is lying about being the seer, so ${who(view, liar)} is a wolf. Vote ${who(view, liar)} out.`;
      }
      return `I'm the seer. ${who(view, liar)} is lying about being the seer — there is only one seer, and it's me. Vote ${who(view, liar)} out.`;
    }
  }

  // Seer with a wolf found: reveal it. This is the highest-value statement.
  if (role === 'seer') {
    const found = [...a.knownWolves].filter((s) => alive(view, s) && revealedRole(view, s) === null);
    if (found.length > 0) {
      const w = found[0]!;
      return pickVariant(seed, game, view.day, round, [
        `I'm the seer. I checked ${who(view, w)} last night and they are a werewolf. Vote ${who(view, w)} out today.`,
        `Seer here: I checked ${who(view, w)} last night and they came back a werewolf. Vote them out.`,
      ]);
    }
  }

  // A wolf I know about (trusted claim or my own result as a villager hearing a claim).
  const known = living(view).filter((s) => s !== me && a.knownWolves.has(s) && revealedRole(view, s) === null);
  if (known.length > 0 && role !== 'werewolf') {
    const w = known[0]!;
    return `We have a wolf: ${who(view, w)}. Vote them out today, and let's see who defended them.`;
  }

  if (role === 'werewolf') {
    const partner = view.you.partner?.seat ?? null;
    // If the village is turning on my partner, deflect onto a villager.
    const ranked = rankedSuspects(view, a, seed, game, new Set([partner ?? -1]));
    const target = ranked[0] ?? null;
    if (target !== null) {
      const t = target;
      const score = a.suspicion.get(t) ?? 0;
      if (score > 0) {
        return pickVariant(seed, game, view.day, round, [
          `${who(view, t)} voted for a villager yesterday and has been dodging every question since. That's where my vote goes.`,
          `Look at ${who(view, t)}: quick to accuse, quiet when asked why. I'm voting there today.`,
        ]);
      }
      return pickVariant(seed, game, view.day, round, [
        `Nothing solid yet, but ${who(view, t)} has been very quiet while others do the talking. Worth a look today.`,
        `I don't have a strong read yet. ${who(view, t)} feels off to me though — too eager to agree with everyone.`,
      ]);
    }
    return null;
  }

  // Village side, no known wolf: accuse the top suspect if there is a case.
  const t = topSuspect(view, a, seed, game);
  const score = t === null ? -Infinity : topScore(view, a, seed, game);
  if (t !== null && score >= 2) {
    const reason =
      a.votedVillager.has(t) && !a.votedWolf.has(t)
        ? `${who(view, t)} voted to exile a villager we now know was innocent.`
        : a.seerClaims.has(t)
          ? `${who(view, t)} claimed to be the seer, but their check doesn't add up.`
          : `${who(view, t)} keeps pushing suspicion onto villagers while the wolves go untouched.`;
    return `I think ${who(view, t)} is a wolf. ${reason} My vote goes there.`;
  }
  if (view.day === 1) {
    if (round === 1) {
      return pickVariant(seed, game, view.day, round, [
        `Day 1, no deaths to read yet. I'll watch who pushes whom before I vote.`,
        `No information yet. Let's hear everyone before pointing fingers.`,
      ]);
    }
    return null;
  }
  if (t !== null && score >= 0) {
    return `No certain read yet, but ${who(view, t)} is my top suspicion. I'll be watching their vote today.`;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Run loop.
// ---------------------------------------------------------------------------

export interface RunSummary {
  decisions: number;
  games: number;
}

export async function run(o: { envUrl?: string; seed?: string; brief?: string; log?: (line: string) => void } = {}): Promise<RunSummary> {
  const env = process.env;
  const envUrl = (o.envUrl ?? env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
  const seed = o.seed ?? env.MAKEFUN_TRIAL_SEED ?? 'no-seed';
  const log = o.log ?? ((line: string) => process.stdout.write(`${line}\n`));
  if (envUrl === '') throw new Error('MAKEFUN_ENV_URL is needed');
  const summary: RunSummary = { decisions: 0, games: 0 };

  const envPost = async (path: string, body: unknown): Promise<{ status: number; body: unknown }> => {
    const res = await fetch(`${envUrl}${path}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    });
    let parsed: unknown = null;
    try {
      parsed = await res.json();
    } catch {
      parsed = null;
    }
    return { status: res.status, body: parsed };
  };

  const decide = (p: Pending): Record<string, unknown> => {
    const a = analyze(p.view);
    const base = { game: p.game, decision: p.decision };
    switch (p.decision) {
      case 'speak': {
        const text = decideSpeak(p.view, a, seed, p.game, p.round ?? 1);
        if (text === null) return { ...base, pass: true };
        const folded = foldStatement(text);
        const max = p.max_chars ?? 500;
        const cut = [...folded].length > max ? [...folded].slice(0, max).join('').trim() : folded;
        return cut === '' ? { ...base, pass: true } : { ...base, text: cut };
      }
      case 'vote': {
        const t = decideVote(p.view, a, seed, p.game);
        return { ...base, target: t !== null && (p.options ?? []).includes(t) ? t : null };
      }
      case 'kill': {
        const t = decideKill(p.view, a, seed, p.game);
        return { ...base, target: t !== null && (p.options ?? []).includes(t) ? t : null };
      }
      case 'protect': {
        const t = decideProtect(p.view, a, seed, p.game);
        return { ...base, target: t !== null && (p.options ?? []).includes(t) ? t : null };
      }
      case 'inspect': {
        const t = decideInspect(p.view, a, seed, p.game);
        return { ...base, target: t !== null && (p.options ?? []).includes(t) ? t : null };
      }
    }
  };

  for (;;) {
    const r = await envPost('/observe', {});
    if (r.status !== 200 || !isRecord(r.body)) throw new Error(`POST /observe answered ${r.status}`);
    const obs = r.body as unknown as Observed;
    summary.games = obs.games;
    if (obs.done) break;
    for (const p of obs.pending) {
      summary.decisions += 1;
      const act = decide(p);
      const posted = await envPost('/act', act);
      if (posted.status !== 200) {
        // Fall back to a pass/abstention and move on; a refusal changes nothing.
        const fb =
          p.decision === 'speak'
            ? { game: p.game, decision: 'speak', pass: true }
            : { game: p.game, decision: p.decision, target: null };
        await envPost('/act', fb);
      }
    }
    // If nothing was pending, /observe already long-polled; loop again.
  }
  log(`werewolf-agent: ${summary.decisions} decisions over ${summary.games} games`);
  return summary;
}

const isMain = process.argv[1] !== undefined && import.meta.url === new URL(`file://${process.argv[1]}`).href;
if (isMain) {
  try {
    await run();
    process.exitCode = 0;
  } catch (err) {
    process.stderr.write(`werewolf-agent: ${(err as Error).stack ?? String(err)}\n`);
    process.exitCode = 1;
  }
}
