"""Producer pipeline that turns open broadcast videos + relay PGN into FEN
annotations (video id + timestamp + crop + FEN), released Kinetics-style.

Stages: relay (PGN -> per-game clock/FEN timeline), video (download + frames),
templates (shot layouts + crop rects), identify (which board + ply from on-screen
text), align (legal-path check + confidence), review (human verify), pipeline
(emit), reconstruct (public rebuild).
"""
