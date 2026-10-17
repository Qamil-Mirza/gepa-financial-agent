# Team workflow (hackathon edition)

1. `git pull` before you start anything.
2. Branch per task: `git checkout -b <name>/<thing>` (e.g. `arjun/planner-loop`).
3. Commit small and often: `[feat]: ...`, `[fix]: ...`, `[docs]: ...`.
4. Push and open a PR into `main`; get one teammate to glance at it.
5. Never commit `.env` or any API key. If you do, revoke the key in the Crusoe Console right away.
6. Last hour: freeze `main`, only demo fixes.
