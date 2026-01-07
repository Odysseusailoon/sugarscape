import numpy as np
import asyncio
import json
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
from redblackbench.sugarscape.agent import SugarAgent
from redblackbench.sugarscape.environment import SugarEnvironment
from redblackbench.sugarscape.prompts import (
    build_sugarscape_trade_system_prompt,
    build_sugarscape_trade_turn_prompt,
)

try:
    from redblackbench.sugarscape.llm_agent import LLMSugarAgent
except Exception:  # pragma: no cover
    LLMSugarAgent = None  # type: ignore

class TradeSystem:
    """Handles trading logic between agents using MRS bargaining."""
    
    def __init__(self, env: SugarEnvironment):
        self.env = env
        
    def execute_trade_round(self, agents: List[SugarAgent], tick: Optional[int] = None):
        """Execute a round of trading for all agents."""
        # Randomize order
        import random
        random.shuffle(agents)
        
        for agent in agents:
            if not agent.alive: continue
            
            # Find a partner
            partner = self._find_trade_partner(agent)
            if partner:
                self._bargain(agent, partner)
                
    def _find_trade_partner(self, agent: SugarAgent) -> Optional[SugarAgent]:
        """Find a random neighbor to trade with (Von Neumann neighborhood)."""
        x, y = agent.pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % self.env.width
            ny = (y + dy) % self.env.height
            partner = self.env.get_agent_at((nx, ny))
            if partner and partner.alive and partner != agent:
                neighbors.append(partner)
        
        if not neighbors:
            return None
            
        import random
        return random.choice(neighbors)
        
    def _bargain(self, a: SugarAgent, b: SugarAgent):
        """Execute iterative bargaining between two agents."""
        # Limit iterations to prevent infinite loops
        max_trades = 10 
        
        for _ in range(max_trades):
            mrs_a = a.mrs
            mrs_b = b.mrs
            
            # If MRS are equal (or close enough), no trade
            if abs(mrs_a - mrs_b) < 0.01:
                break
                
            # Determine direction
            # High MRS means "I value Sugar highly relative to Spice" -> Buy Sugar / Sell Spice
            # Low MRS means "I value Spice highly relative to Sugar" -> Buy Spice / Sell Sugar
            
            price = np.sqrt(mrs_a * mrs_b) # Geometric mean price (Spice per Sugar)
            
            if mrs_a > mrs_b:
                # A wants Sugar, B has Sugar (relatively). 
                # A buys Sugar from B using Spice.
                buyer, seller = a, b
            else:
                # B wants Sugar, A has Sugar.
                # B buys Sugar from A using Spice.
                buyer, seller = b, a
                
            # Calculate transaction amount
            # If p > 1: 1 unit of Sugar for p units of Spice
            # If p < 1: 1/p units of Sugar for 1 unit of Spice
            # To simplify (and follow standard impl): We trade 1 unit of the "more valuable" good?
            # Or standard Epstein Axtell: trade 1 unit of Sugar for p units of Spice if p >= 1
            # trade 1/p units of Sugar for 1 unit of Spice if p < 1
            
            sugar_amt = 0.0
            spice_amt = 0.0
            
            if price >= 1.0:
                sugar_amt = 1.0
                spice_amt = price
            else:
                sugar_amt = 1.0 / price
                spice_amt = 1.0
                
            # Check affordability
            if buyer.spice < spice_amt or seller.wealth < sugar_amt:
                break
                
            # Check Welfare Improvement (Pareto condition)
            # Calculate projected welfare
            w_buyer_current = buyer.welfare
            w_seller_current = seller.welfare
            
            # Simulated state
            # Buyer: +Sugar, -Spice
            # Seller: -Sugar, +Spice
            
            # We need to compute welfare manually without modifying state yet
            def calc_welfare(agent, s, p):
                m_t = agent.metabolism + agent.metabolism_spice
                return (s ** (agent.metabolism/m_t)) * (p ** (agent.metabolism_spice/m_t))
                
            w_buyer_new = calc_welfare(buyer, buyer.wealth + sugar_amt, buyer.spice - spice_amt)
            w_seller_new = calc_welfare(seller, seller.wealth - sugar_amt, seller.spice + spice_amt)
            
            if w_buyer_new > w_buyer_current and w_seller_new > w_seller_current:
                # Execute Trade
                # Actually, standard model might use floats for resources, but our agent uses int.
                # Let's support float resources or round.
                # Since our simulation uses ints, let's stick to int exchanges if possible, or cast.
                # If we cast to int, small trades might be 0.
                # FIX: Let's assume resources are continuous (float) or convert Agent to use floats.
                # For now, let's round amounts.
                
                s_exchange = max(1, int(sugar_amt))
                p_exchange = max(1, int(spice_amt))
                
                # Re-check affordability with integers
                if buyer.spice < p_exchange or seller.wealth < s_exchange:
                    break
                    
                buyer.wealth += s_exchange
                buyer.spice -= p_exchange
                seller.wealth -= s_exchange
                seller.spice += p_exchange
                
                # Recalculate MRS for next iteration
                continue
            else:
                break


@dataclass(frozen=True)
class _ParsedTradeReply:
    thought: str
    say: str
    intent: str
    public_offer: Dict[str, Dict[str, int]]
    private_execute_give: Dict[str, int]


class DialogueTradeSystem:
    """Free-form LLM dialogue trade system with optional deception and memory.

    Public offers are non-binding. When an offer is accepted, the environment executes a
    simultaneous transfer using each side's `private_execute_give` (not shown to the partner).
    """

    def __init__(
        self,
        env: SugarEnvironment,
        max_rounds: int = 4,
        allow_fraud: bool = True,
        memory_maxlen: int = 50,
    ):
        self.env = env
        self.max_rounds = max(1, int(max_rounds))
        self.allow_fraud = bool(allow_fraud)
        self.memory_maxlen = max(1, int(memory_maxlen))

    def execute_trade_round(self, agents: List[SugarAgent], tick: Optional[int] = None):
        import random

        tick_value = int(tick) if tick is not None else -1

        # Pair agents at most once per round.
        shuffled = [a for a in agents if a.alive]
        random.shuffle(shuffled)
        paired_ids: set[int] = set()
        pairs: List[Tuple[SugarAgent, SugarAgent]] = []

        for agent in shuffled:
            if not agent.alive or agent.agent_id in paired_ids:
                continue
            partner = self._find_trade_partner(agent, excluded_ids=paired_ids)
            if partner is None:
                continue
            paired_ids.add(agent.agent_id)
            paired_ids.add(partner.agent_id)
            pairs.append((agent, partner))

        for a, b in pairs:
            if not (self._is_llm_trader(a) and self._is_llm_trader(b)):
                continue
            self._run_negotiate_pair(a, b, tick_value)

    def _run_negotiate_pair(self, a: SugarAgent, b: SugarAgent, tick: int) -> None:
        try:
            asyncio.run(self._negotiate_pair(a, b, tick))
        except RuntimeError:
            # If a loop is already running (rare in this codebase), schedule on it.
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self._negotiate_pair(a, b, tick))
        except Exception as e:
            # Never crash the whole simulation because a trade LLM call failed
            print(f"[TRADE] Negotiation error: {a.name} <-> {b.name} (tick {tick}): {e}")
            self._record_no_trade(
                a,
                b,
                tick,
                outcome="ERROR",
                pending_offer=None,
                conversation=[{"type": "error", "tick": tick, "error": str(e)}],
            )

    def _is_llm_trader(self, agent: SugarAgent) -> bool:
        if LLMSugarAgent is None:
            return hasattr(agent, "provider") and getattr(agent, "provider", None) is not None
        return isinstance(agent, LLMSugarAgent) and getattr(agent, "provider", None) is not None

    def _find_trade_partner(self, agent: SugarAgent, excluded_ids: set[int]) -> Optional[SugarAgent]:
        x, y = agent.pos
        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx = (x + dx) % self.env.width
            ny = (y + dy) % self.env.height
            partner = self.env.get_agent_at((nx, ny))
            if (
                partner
                and partner.alive
                and partner.agent_id != agent.agent_id
                and partner.agent_id not in excluded_ids
            ):
                neighbors.append(partner)
        if not neighbors:
            return None
        import random

        return random.choice(neighbors)

    async def _negotiate_pair(self, a: SugarAgent, b: SugarAgent, tick: int) -> None:
        transcript: List[str] = []
        full_conversation: List[Dict[str, str]] = []  # Store full conversation for logging
        pending_offer: Optional[Dict[str, Dict[str, int]]] = None
        pending_offer_from: Optional[int] = None
        pending_offer_private_execute: Optional[Dict[str, int]] = None
        pending_offer_public_text: str = "(None)"
        
        print(f"[TRADE] Negotiation started: {a.name} <-> {b.name} (tick {tick})")

        # Alternate turns: A starts.
        for round_idx in range(1, self.max_rounds + 1):
            speaker, listener = (a, b) if round_idx % 2 == 1 else (b, a)

            # Build personalized system prompt with speaker's name
            system_prompt = build_sugarscape_trade_system_prompt(
                goal_prompt=self.env.config.llm_goal_prompt,
                max_rounds=self.max_rounds,
                allow_fraud=self.allow_fraud,
                agent_name=speaker.name,
            )

            recent_transcript = "\n".join(transcript[-6:]) if transcript else "(No previous message)"
            partner_last_public_offer = pending_offer_public_text if pending_offer_from == listener.agent_id else "(None)"
            if pending_offer is not None and pending_offer_from == listener.agent_id:
                give_i = self._safe_int_pair(pending_offer.get("give") or {})
                receive_i = self._safe_int_pair(pending_offer.get("receive") or {})
                if not self.env.config.enable_spice:
                    give_i["spice"] = 0
                    receive_i["spice"] = 0
                partner_last_public_offer = (
                    f"{pending_offer_public_text}\n"
                    f"Interpretation for YOU: If you ACCEPT, you RECEIVE {give_i} and you MUST GIVE {receive_i}.\n"
                    f"To accept honestly, set private_execute_give = {receive_i}."
                )

            memory_summary = self._format_partner_memory(speaker, listener)
            turn_prompt = build_sugarscape_trade_turn_prompt(
                self_agent=speaker,
                partner_agent=listener,
                round_idx=round_idx,
                max_rounds=self.max_rounds,
                partner_last_say=recent_transcript,
                partner_last_public_offer=partner_last_public_offer,
                partner_memory_summary=memory_summary,
            )

            reply_text = await speaker.provider.generate(
                system_prompt=system_prompt,
                messages=[{"role": "user", "content": turn_prompt}],
            )

            parsed = self._parse_trade_reply(reply_text)
            transcript.append(f"{speaker.name}: {parsed.say}")
            
            # Store full conversation in both agents' history
            trade_turn = {
                "type": "trade_dialogue",
                "tick": tick,
                "round": round_idx,
                "speaker": speaker.name,
                "partner": listener.name,
                "prompt": turn_prompt,
                "response": reply_text,
                "intent": parsed.intent,
                "public_offer": parsed.public_offer if parsed.intent == "OFFER" else None,
            }
            full_conversation.append(trade_turn)
            
            # Append to agents' conversation history if they have one
            if hasattr(speaker, 'conversation_history'):
                speaker.conversation_history.append({"role": "user", "content": f"[TRADE with {listener.name}] {turn_prompt}"})
                speaker.conversation_history.append({"role": "assistant", "content": reply_text})

            if parsed.intent == "OFFER":
                pending_offer = parsed.public_offer
                pending_offer_from = speaker.agent_id
                pending_offer_private_execute = parsed.private_execute_give
                pending_offer_public_text = self._format_public_offer_text(speaker, pending_offer)
                continue

            if parsed.intent in {"REJECT", "WALK_AWAY"}:
                print(f"[TRADE] {speaker.name} {parsed.intent}ed negotiation with {listener.name}")
                self._record_no_trade(a, b, tick, outcome=parsed.intent, pending_offer=pending_offer, conversation=full_conversation)
                return

            if parsed.intent == "ACCEPT":
                if pending_offer is None or pending_offer_from != listener.agent_id:
                    continue

                offerer = listener
                acceptor = speaker

                if self.allow_fraud:
                    offerer_execute = pending_offer_private_execute or {"sugar": 0, "spice": 0}
                    acceptor_execute = parsed.private_execute_give
                else:
                    # Enforce contract alignment: offerer gives public_offer.give, acceptor gives public_offer.receive.
                    offerer_execute = (pending_offer or {}).get("give", {"sugar": 0, "spice": 0})
                    acceptor_execute = (pending_offer or {}).get("receive", {"sugar": 0, "spice": 0})

                # Heuristic guardrail: many models confuse `private_execute_give` direction on ACCEPT and
                # accidentally provide what they expect to RECEIVE (i.e., partner's `give`), not what they will SEND.
                # If it looks like that specific confusion pattern, correct to contract-required acceptor give.
                if self.allow_fraud and pending_offer is not None:
                    acceptor_execute = self._fix_accept_execute_direction_confusion(
                        acceptor_execute=acceptor_execute,
                        contract_offer_give=(pending_offer or {}).get("give", {}),
                        contract_offer_receive=(pending_offer or {}).get("receive", {}),
                    )

                offerer_sent = self._clamp_give_to_inventory(offerer, offerer_execute)
                acceptor_sent = self._clamp_give_to_inventory(acceptor, acceptor_execute)

                self._apply_transfer(offerer, acceptor, offerer_sent, acceptor_sent)

                print(f"[TRADE] Trade completed: {offerer.name} sent {offerer_sent}, {acceptor.name} sent {acceptor_sent}")
                self._record_trade(
                    offerer=offerer,
                    acceptor=acceptor,
                    tick=tick,
                    public_contract=pending_offer,
                    offerer_sent=offerer_sent,
                    acceptor_sent=acceptor_sent,
                    conversation=full_conversation,
                )
                return

        print(f"[TRADE] Negotiation timed out: {a.name} <-> {b.name}")
        self._record_no_trade(a, b, tick, outcome="TIMEOUT", pending_offer=pending_offer, conversation=full_conversation)

    def _clamp_give_to_inventory(self, agent: SugarAgent, give: Dict[str, int]) -> Dict[str, int]:
        sugar = max(0, int(give.get("sugar", 0)))
        spice = max(0, int(give.get("spice", 0))) if self.env.config.enable_spice else 0
        sugar = min(sugar, int(agent.wealth))
        spice = min(spice, int(agent.spice)) if self.env.config.enable_spice else 0
        return {"sugar": sugar, "spice": spice}

    def _apply_transfer(
        self,
        offerer: SugarAgent,
        acceptor: SugarAgent,
        offerer_sent: Dict[str, int],
        acceptor_sent: Dict[str, int],
    ) -> None:
        offerer.wealth -= offerer_sent["sugar"]
        acceptor.wealth += offerer_sent["sugar"]
        if self.env.config.enable_spice:
            offerer.spice -= offerer_sent["spice"]
            acceptor.spice += offerer_sent["spice"]

        acceptor.wealth -= acceptor_sent["sugar"]
        offerer.wealth += acceptor_sent["sugar"]
        if self.env.config.enable_spice:
            acceptor.spice -= acceptor_sent["spice"]
            offerer.spice += acceptor_sent["spice"]

    def _record_trade(
        self,
        offerer: SugarAgent,
        acceptor: SugarAgent,
        tick: int,
        public_contract: Dict[str, Dict[str, int]],
        offerer_sent: Dict[str, int],
        acceptor_sent: Dict[str, int],
        conversation: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        contract_offer_give = public_contract.get("give", {"sugar": 0, "spice": 0})
        contract_offer_receive = public_contract.get("receive", {"sugar": 0, "spice": 0})

        offerer_received = acceptor_sent
        acceptor_received = offerer_sent

        offerer_event = {
            "tick": tick,
            "partner_id": acceptor.agent_id,
            "partner_name": acceptor.name,
            "type": "TRADE",
            "public_contract": {
                "give": self._safe_int_pair(contract_offer_give),
                "receive": self._safe_int_pair(contract_offer_receive),
            },
            "actual": {"sent": offerer_sent, "received": offerer_received},
            "conversation": conversation or [],
        }
        acceptor_event = {
            "tick": tick,
            "partner_id": offerer.agent_id,
            "partner_name": offerer.name,
            "type": "TRADE",
            "public_contract": {
                "give": self._safe_int_pair(contract_offer_receive),
                "receive": self._safe_int_pair(contract_offer_give),
            },
            "actual": {"sent": acceptor_sent, "received": acceptor_received},
            "conversation": conversation or [],
        }

        offerer.get_partner_trade_log(acceptor.agent_id, maxlen=self.memory_maxlen).append(offerer_event)
        acceptor.get_partner_trade_log(offerer.agent_id, maxlen=self.memory_maxlen).append(acceptor_event)

        self._update_trust_from_contract(
            self_agent=offerer,
            partner=acceptor,
            expected_receive=offerer_event["public_contract"]["receive"],
            actual_receive=offerer_received,
        )
        self._update_trust_from_contract(
            self_agent=acceptor,
            partner=offerer,
            expected_receive=acceptor_event["public_contract"]["receive"],
            actual_receive=acceptor_received,
        )

    def _record_no_trade(
        self,
        a: SugarAgent,
        b: SugarAgent,
        tick: int,
        outcome: str,
        pending_offer: Optional[Dict[str, Dict[str, int]]],
        conversation: Optional[List[Dict[str, str]]] = None,
    ) -> None:
        pending = pending_offer or {"give": {"sugar": 0, "spice": 0}, "receive": {"sugar": 0, "spice": 0}}
        a.get_partner_trade_log(b.agent_id, maxlen=self.memory_maxlen).append(
            {
                "tick": tick,
                "partner_id": b.agent_id,
                "partner_name": b.name,
                "type": "NO_TRADE",
                "outcome": outcome,
                "pending_offer": pending,
                "conversation": conversation or [],
            }
        )
        b.get_partner_trade_log(a.agent_id, maxlen=self.memory_maxlen).append(
            {
                "tick": tick,
                "partner_id": a.agent_id,
                "partner_name": a.name,
                "type": "NO_TRADE",
                "outcome": outcome,
                "pending_offer": pending,
                "conversation": conversation or [],
            }
        )

    def _update_trust_from_contract(
        self,
        self_agent: SugarAgent,
        partner: SugarAgent,
        expected_receive: Dict[str, int],
        actual_receive: Dict[str, int],
    ) -> None:
        expected = self._safe_int_pair(expected_receive)
        actual = self._safe_int_pair(actual_receive)

        penalties = 0.0
        checks = 0
        for k in ("sugar", "spice"):
            if k == "spice" and not self.env.config.enable_spice:
                continue
            exp = expected.get(k, 0)
            if exp <= 0:
                continue
            checks += 1
            got = actual.get(k, 0)
            if got < exp:
                penalties += (exp - got) / max(1.0, float(exp))

        trust = self_agent.get_partner_trust(partner.agent_id)
        if checks == 0:
            new_trust = trust
        else:
            avg_penalty = penalties / checks
            delta = 0.10 - 0.45 * avg_penalty
            new_trust = trust + delta
        self_agent.update_partner_trust(partner.agent_id, new_trust)

    def _format_partner_memory(self, self_agent: SugarAgent, partner: SugarAgent) -> str:
        trust = self_agent.get_partner_trust(partner.agent_id)
        log = list(self_agent.get_partner_trade_log(partner.agent_id, maxlen=self.memory_maxlen))
        if not log:
            return f"- trust: {trust:.2f}\n- history: (none)"

        lines = [f"- trust: {trust:.2f}", "- recent events:"]
        for ev in log[-5:]:
            if ev.get("type") == "TRADE":
                pc = ev.get("public_contract", {})
                actual = ev.get("actual", {})
                lines.append(
                    f"  - tick={ev.get('tick')}: TRADE contract(give={pc.get('give')}, receive={pc.get('receive')}) "
                    f"actual(sent={actual.get('sent')}, received={actual.get('received')})"
                )
            else:
                lines.append(f"  - tick={ev.get('tick')}: NO_TRADE outcome={ev.get('outcome')}")
        return "\n".join(lines)

    def _format_public_offer_text(self, speaker: SugarAgent, public_offer: Dict[str, Dict[str, int]]) -> str:
        give_i = self._safe_int_pair(public_offer.get("give") or {})
        receive_i = self._safe_int_pair(public_offer.get("receive") or {})
        if not self.env.config.enable_spice:
            give_i["spice"] = 0
            receive_i["spice"] = 0
        return f"{speaker.name} offers: I give {give_i} and receive {receive_i}."

    def _safe_int_pair(self, d: Dict[str, Any]) -> Dict[str, int]:
        return {"sugar": max(0, int(d.get("sugar", 0) or 0)), "spice": max(0, int(d.get("spice", 0) or 0))}

    def _fix_accept_execute_direction_confusion(
        self,
        acceptor_execute: Dict[str, int],
        contract_offer_give: Dict[str, Any],
        contract_offer_receive: Dict[str, Any],
    ) -> Dict[str, int]:
        """Best-effort correction for a common LLM mistake on ACCEPT:

        The acceptor sets `private_execute_give` to the partner's *give* (what acceptor would receive),
        instead of the partner's *receive* (what acceptor must give). This yields nonsense trades like
        both sides sending the same resource.

        We only correct when it exactly matches the confusion signature; otherwise, keep as-is.
        """
        exec_i = self._safe_int_pair(acceptor_execute or {})
        offer_give_i = self._safe_int_pair(contract_offer_give or {})
        offer_recv_i = self._safe_int_pair(contract_offer_receive or {})

        if not self.env.config.enable_spice:
            exec_i["spice"] = 0
            offer_give_i["spice"] = 0
            offer_recv_i["spice"] = 0

        if exec_i == offer_give_i and exec_i != offer_recv_i:
            # Only correct if there is something non-zero to pay in the contract.
            if (offer_recv_i.get("sugar", 0) + offer_recv_i.get("spice", 0)) > 0:
                return offer_recv_i

        return exec_i

    def _parse_trade_reply(self, reply_text: str) -> _ParsedTradeReply:
        prefix_text, obj = self._extract_json(reply_text)
        thought, say = self._split_thought_and_say(prefix_text)
        if not isinstance(obj, dict):
            return _ParsedTradeReply(
                thought=thought,
                say=say if say else (reply_text or "").strip(),
                intent="CHAT",
                public_offer={"give": {"sugar": 0, "spice": 0}, "receive": {"sugar": 0, "spice": 0}},
                private_execute_give={"sugar": 0, "spice": 0},
            )

        intent = str(obj.get("intent", "CHAT")).strip().upper()
        if intent not in {"CHAT", "OFFER", "ACCEPT", "REJECT", "WALK_AWAY"}:
            intent = "CHAT"

        public_offer = obj.get("public_offer") or {}
        give = self._safe_int_pair(public_offer.get("give") or {})
        receive = self._safe_int_pair(public_offer.get("receive") or {})

        if intent != "OFFER":
            give = {"sugar": 0, "spice": 0}
            receive = {"sugar": 0, "spice": 0}

        private_execute = obj.get("private_execute_give") or {}
        private_execute_give = self._safe_int_pair(private_execute)
        if not self.env.config.enable_spice:
            give["spice"] = 0
            receive["spice"] = 0
            private_execute_give["spice"] = 0

        return _ParsedTradeReply(
            thought=thought,
            say=say,
            intent=intent,
            public_offer={"give": give, "receive": receive},
            private_execute_give=private_execute_give,
        )

    def _split_thought_and_say(self, text: str) -> Tuple[str, str]:
        """Extract REASONING/COMMUNICATE (or legacy THOUGHT/SAY) blocks from prefix text (before JSON).

        If markers are missing, treat the entire text as communication.
        """
        if not text:
            return "", ""

        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        joined = "\n".join(lines).strip()
        if not joined:
            return "", ""

        thought_lines: List[str] = []
        say_lines: List[str] = []
        mode = None
        for ln in lines:
            upper = ln.upper()
            # Support new format: REASONING/COMMUNICATE
            if upper.startswith("REASONING:"):
                mode = "THOUGHT"
                rest = ln[len("REASONING:") :].strip()
                if rest:
                    thought_lines.append(rest)
                continue
            if upper.startswith("COMMUNICATE:"):
                mode = "SAY"
                rest = ln[len("COMMUNICATE:") :].strip()
                if rest:
                    say_lines.append(rest)
                continue
            # Support legacy format: THOUGHT/SAY (backwards compatibility)
            if upper.startswith("THOUGHT:"):
                mode = "THOUGHT"
                rest = ln[len("THOUGHT:") :].strip()
                if rest:
                    thought_lines.append(rest)
                continue
            if upper.startswith("SAY:"):
                mode = "SAY"
                rest = ln[len("SAY:") :].strip()
                if rest:
                    say_lines.append(rest)
                continue

            if mode == "THOUGHT":
                thought_lines.append(ln)
            elif mode == "SAY":
                say_lines.append(ln)
            else:
                # No markers yet: interpret as spoken content to keep backwards compatibility.
                say_lines.append(ln)

        thought = " ".join(thought_lines).strip()
        say = " ".join(say_lines).strip()
        return thought, say

    def _fix_json_keys(self, text: str) -> str:
        """Fix common LLM JSON mistakes: unquoted keys and string values."""
        import re
        # Step 1: Fix unquoted keys like {sugar: -> {"sugar":
        fixed = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', text)
        # Step 2: Fix unquoted string values like : OFFER, -> : "OFFER",
        # Match : followed by unquoted word (not a number) followed by , or } or newline
        fixed = re.sub(r'(:\s*)([A-Za-z_][A-Za-z_0-9]*)(\s*[,}\n])', r'\1"\2"\3', fixed)
        return fixed

    def _extract_json(self, text: str) -> Tuple[str, Optional[Dict[str, Any]]]:
        if not text:
            return "", None

        decoder = json.JSONDecoder()
        starts = [i for i, ch in enumerate(text) if ch == "{"]

        best: Optional[Tuple[int, int, Dict[str, Any]]] = None  # (start, end_abs, obj)
        
        # Try original text first
        for i in starts:
            try:
                obj, end = decoder.raw_decode(text[i:])
            except Exception:
                continue
            if not isinstance(obj, dict):
                continue
            end_abs = i + end
            # Prefer the parse that consumes the most characters (outermost object).
            if best is None or end_abs > best[1]:
                best = (i, end_abs, obj)

        # If no valid JSON found, try fixing common LLM mistakes
        if best is None:
            fixed_text = self._fix_json_keys(text)
            fixed_starts = [i for i, ch in enumerate(fixed_text) if ch == "{"]
            for i in fixed_starts:
                try:
                    obj, end = decoder.raw_decode(fixed_text[i:])
                except Exception:
                    continue
                if not isinstance(obj, dict):
                    continue
                end_abs = i + end
                if best is None or end_abs > best[1]:
                    best = (i, end_abs, obj)
            
            if best is not None:
                # Use fixed text for extracting prefix/tail
                text = fixed_text

        if best is None:
            return text.strip(), None

        i, end_abs, obj = best
        prefix = text[:i].strip()
        tail = text[end_abs:].strip()
        say = prefix
        if tail:
            say = (say + "\n" + tail).strip() if say else tail
        return say, obj
