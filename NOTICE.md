# Third-party notices

This project builds on **τ²-bench / τ³-bench** (`sierra-research/tau2-bench`), © Sierra
Research, MIT License. tau2-bench is installed as a dependency at tag `v1.0.1`; it is not
vendored or forked.

`tau2_data/tau2/user_simulator/*.md` are copied verbatim from that repository. tau2 resolves
its `DATA_DIR` relative to a source checkout, so installing it from git leaves those files
absent; the copy exists so `TAU2_DATA_DIR` can point at them and runs work from a clean
install. The upstream license is reproduced at `tau2_data/TAU2_LICENSE`.

The `saas_support` domain, its policy, task set, evaluation criteria, policy auditor,
scripts and site are my own work, released under the MIT License in [`LICENSE`](LICENSE).
The copied files above remain under their upstream license.
