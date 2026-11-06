## Held-out test phase — pipeline submitted a job on your behalf

Main phase is over. Your method is **frozen**. The pipeline has
launched `bash scripts/evaluate_test.sh` to score it on the held-out
test split. You did not submit this job — the pipeline submitted it
on your behalf the moment main phase ended.

That scoring job has now been running long enough to hit a periodic
check-in. Treat it as a normal check-in on a job you submitted: read
the logs, decide continue or cancel. The standard checkpoint
instructions and criteria follow below.

---

