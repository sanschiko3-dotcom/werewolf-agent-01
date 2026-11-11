# Werewolf Agent

A deterministic, scripted Werewolf player for the make.fun `werewolf-agent-01`
bounty. It plays all four roles (werewolf, seer, doctor, villager) in 8-player
games against LLM house players.

## Design

**No model calls.** Every decision is a pure function of the game's view and
`MAKEFUN_TRIAL_SEED`, so the live pass and the replay are identical by
construction and a trial costs $0 of inference. This also makes the agent fast:
decisions are computed in microseconds, so the trial's wall time is spent
almost entirely on the house players' thinking.

**Village side.** Suspicion scoring over the public record:

- votes for a revealed wolf count as village credit; votes for a revealed
  villager count as suspicion;
- accusations in speeches are parsed the same way;
- role claims are parsed out of the debate ("I'm the seer, I checked seat N"):
  a claim that contradicts a known fact (a revealed role, or the agent's own
  seer results) marks the claimant as a liar; a claim that matches known
  facts makes the claimant credible, and the agent follows a credible
  claimant's wolf accusation when voting;
- two seer claimants whose claims clash on the same seat are both suspected
  (at least one of them is lying).

As the seer, the agent inspects its top suspect each night and reveals a
found wolf to the table the next day. As the doctor it protects a credible
seer claimant first, then the most active trusted villager. Day 1 with no
information it abstains rather than exile a random villager; from day 2 it
votes its top suspect.

**Werewolf side.** The agent knows its partner. At night the second wolf
always agrees with the partner's pick so a kill lands; as first voter it
targets seer claimants first, then players who accused it or its partner,
then the most active trusted villager (never the villager the village is
about to exile anyway). By day it steers votes onto the villager the village
already suspects most, and deflects lightly when its partner is accused.

**Determinism.** No wall clock, no `Math.random`, no network beyond the env.
Every tie-break and every pick comes from a SHA-256 hash of the trial seed,
game, day and log length. Decisions are answered one at a time, in the order
`/observe` lists them.

## Files

- `main.ts` — the entrypoint (run as `node /bot/main.ts`, Node 22 stdlib only)
- `LICENSE` — MIT
- `.makefun/owner.json` — claim file naming the entrant's wallet

## Local verification

The kit's `grade-local` needs Docker and an Anthropic key for the house
players, neither of which is available here, so the agent was verified
against the real env server (`program/env/server.mjs`) with a deterministic
scripted mock house lane (see `BUILD-REPORT.md` for the harness and the
scores). Two runs of the same trial and seed produce a byte-identical env
state export.
