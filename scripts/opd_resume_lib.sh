
r4_save_steps() {
  local total=$1 spe=$2 every=${3:-$2} off=${4:-$(( $2 / 2 ))} s out=""
  for (( s = off; s < total; s += every )); do
    (( s % spe == 0 )) && continue
    out="${out:+${out}x}$s"
  done
  printf '%s\n' "$out"
}

_r4_tracker() { cat "$1/latest_checkpointed_iteration.txt" 2>/dev/null || echo 0; }

r4_resume_point() {
  local ckpt=$1 n=$2 spe=$3 best="" gs s t
  t=$(_r4_tracker "$ckpt")
  for gs in "$ckpt"/global_step_*; do
    [ -d "$gs/actor" ] || continue
    s=${gs##*global_step_}
    case "$s" in ''|*[!0-9]*) continue ;; esac
    (( s % spe != 0 )) || continue
    (( s <= t )) || continue
    [ -f "$gs/data.pt" ] || continue
    [ "$(ls "$gs/actor"/model_world_size_"${n}"_rank_*.pt 2>/dev/null | wc -l)" = "$n" ] || continue
    [ "$(ls "$gs/actor"/optim_world_size_"${n}"_rank_*.pt 2>/dev/null | wc -l)" = "$n" ] || continue
    [ "$(ls "$gs/actor"/extra_state_world_size_"${n}"_rank_*.pt 2>/dev/null | wc -l)" = "$n" ] || continue
    if [ -z "$best" ] || (( s > best )); then best=$s; fi
  done
  printf '%s\n' "$best"
}

r4_prepare_resume() {
  local ckpt=$1 rdir=$2 k=$3
  python3 - "$ckpt" "$rdir" "$k" <<'PYEOF'
import json, os, re, shutil, sys
ckpt, rdir, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
for name in ("erropd_rollouts.jsonl", "erropd_step_metrics.jsonl"):
    p = f"{ckpt}/{name}"
    if os.path.exists(p):
        keep, drop = [], 0
        for l in open(p):
            if not l.strip():
                continue
            try:
                ok = json.loads(l).get("step", 0) <= k
            except Exception:
                ok = False
            if ok:
                keep.append(l if l.endswith("\n") else l + "\n")
            else:
                drop += 1
        with open(p + ".tmp", "w") as fh:
            fh.writelines(keep)
        os.replace(p + ".tmp", p)
        print(f"[r4_resume] {name}: kept {len(keep)} rows with step <= {k}, dropped {drop}")
for d in sorted(os.listdir(ckpt)):
    m = re.fullmatch(r"global_step_(\d+)", d)
    if m and int(m.group(1)) > k:
        shutil.rmtree(f"{ckpt}/{d}", ignore_errors=True)
        print(f"[r4_resume] removed {d} (after the resume point; trained again)")
if rdir and os.path.isdir(rdir):
    for f in os.listdir(rdir):
        m = re.fullmatch(r"(\d+)\.jsonl", f)
        if m and int(m.group(1)) > k:
            os.remove(f"{rdir}/{f}")
with open(f"{ckpt}/latest_checkpointed_iteration.txt.tmp", "w") as fh:
    fh.write(str(k))
os.replace(f"{ckpt}/latest_checkpointed_iteration.txt.tmp", f"{ckpt}/latest_checkpointed_iteration.txt")
print(f"[r4_resume] tracker -> {k}")
PYEOF
}

_r4_is_eval() { [[ " $2 " == *" $1 "* ]]; }

_r4_keeper_pass() {
  local ckpt=$1 spe=$2 evals=$3 t gs s newest=""
  t=$(_r4_tracker "$ckpt")
  for gs in "$ckpt"/global_step_*; do
    [ -d "$gs" ] || continue
    s=${gs##*global_step_}
    case "$s" in ''|*[!0-9]*) continue ;; esac
    (( s <= t )) || continue
    if (( s % spe != 0 )) && ! _r4_is_eval "$s" "$evals" && [ -f "$gs/data.pt" ]; then
      if [ -z "$newest" ] || (( s > newest )); then newest=$s; fi
    fi
  done
  for gs in "$ckpt"/global_step_*; do
    [ -d "$gs" ] || continue
    s=${gs##*global_step_}
    case "$s" in ''|*[!0-9]*) continue ;; esac
    (( s <= t )) || continue
    if _r4_is_eval "$s" "$evals"; then
      (( s % spe == 0 )) || continue
      if [ -f "$gs/data.pt" ] || ls "$gs"/actor/optim_world_size_*.pt >/dev/null 2>&1; then
        rm -f "$gs"/actor/optim_world_size_*.pt "$gs"/actor/extra_state_world_size_*.pt "$gs/data.pt"
        echo "[r4_keeper] $(date -u +%FT%TZ) step $s: optimizer/extra/dataloader state removed (never a resume point)"
      fi
      if [ -d "$gs/actor" ] && [ ! -f "$gs/hf/.merge_complete" ] && declare -F r4_merge_step >/dev/null; then
        echo "[r4_keeper] $(date -u +%FT%TZ) step $s: background merge"
        if r4_merge_step "$s" && [ -f "$gs/hf/.merge_complete" ]; then
          rm -rf "$gs/actor"
          echo "[r4_keeper] $(date -u +%FT%TZ) step $s: merged, shards removed"
        else
          echo "[r4_keeper] $(date -u +%FT%TZ) step $s: background merge failed; shards kept for the final pass"
        fi
      fi
    elif (( s % spe == 0 )); then
      rm -rf "$gs"; echo "[r4_keeper] $(date -u +%FT%TZ) step $s: boundary checkpoint that is not evaluated removed"
    elif [ -n "$newest" ] && (( s < newest )); then
      rm -rf "$gs"; echo "[r4_keeper] $(date -u +%FT%TZ) step $s: superseded resume point removed (newest $newest)"
    fi
  done
}

r4_keeper() {
  local ckpt=$1 spe=$2 stop=$3 evals=$4 owner=${5:-$$}
  set +e
  while [ ! -f "$stop" ] && kill -0 "$owner" 2>/dev/null; do
    _r4_keeper_pass "$ckpt" "$spe" "$evals"
    local i; for i in 1 2 3 4 5 6; do [ -f "$stop" ] && break; sleep 5; done
  done
}

r4_final_cleanup() {
  local ckpt=$1 evals=$2 gs s
  for gs in "$ckpt"/global_step_*; do
    [ -d "$gs" ] || continue
    s=${gs##*global_step_}
    if _r4_is_eval "$s" "$evals"; then
      rm -f "$gs"/actor/optim_world_size_*.pt "$gs"/actor/extra_state_world_size_*.pt "$gs/data.pt"
      continue
    fi
    rm -rf "$gs"; echo "[r4_cleanup] removed resume-only checkpoint global_step_$s"
  done
}

r4_resumed_kv() {
  local f=$1
  [ -s "$f" ] && printf '"resumed_from":"%s",' "$(grep -o 'global_step_[0-9]*' "$f" | sed 's/global_step_//' | paste -sd, -)"
  return 0
}
