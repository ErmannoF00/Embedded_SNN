# Release verification

The following checks were executed on the cleaned v0.4 source in a separate Linux environment on 6 November 2026:

- `PYTHONPATH=. python -m pytest -q`: **25 tests passed**.
- `cmake -S cpp -B cpp/build` and `cmake --build cpp/build`: host C++ packet parser built.
- GCC C++17 `-fsanitize=address,undefined` parser: one valid 250×8 float32 frame decoded, no reported sanitizer errors.
- Guarded upgrade from unchanged v0.3 installed sources in a simulated Git repository: preflight, backup, 36 source updates, 3 additions, 12 retired paths, preservation of unrelated EEG/checkpoint files, clean post-upgrade C++ build and 25 tests passed.
- A repeated upgrade refused to overwrite the already changed source.

These checks do not prove MCU flashability, subject-independent decoding accuracy or physical energy efficiency. Real EEG experimental numbers in `RESULTS.md` originate from the user's WSL execution logs and are not reproduced in this release container.
