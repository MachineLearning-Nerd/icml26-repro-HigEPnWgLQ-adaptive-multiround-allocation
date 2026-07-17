# Negative controls


---
<!-- trackio-cell
{"type": "markdown", "id": "cell_0abf344ad605", "created_at": "2026-07-17T04:36:43+00:00", "title": "Controls"}
-->
Controls fail closed: malformed/non-normalized PMFs raise `ValueError`; fabricated increasing survival tails are rejected because they cannot arise from a probability distribution; direct PMF enumeration checks the marginal-tail identity; homogeneous exact models force every robustness term and actual loss to zero; and deliberately noisy models produce nonzero policy loss.


---
<!-- trackio-cell
{"type": "code", "id": "cell_6926c6b53e8d", "created_at": "2026-07-17T04:36:45+00:00", "title": "Adversarial and regression tests", "command": ["pytest", "-q"], "exit_code": 0, "duration_s": 1.614}
-->
````bash
$ pytest -q
````

exit 0 · 1.6s


````output
...................                                                      [100%]
19 passed in 1.36s

````
