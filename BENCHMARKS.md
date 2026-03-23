SanMiguel, max bounces 1, release, average of two runs

`cargo run --release -- -b 1 -m <mode>`

| Commit  | Packet4      | Packet4x2 (formerly Packet4) | Packet8   | Stream       | Comments                    |
| ------- | ------------ | ---------------------------- | --------- | ------------ | --------------------------- |
| 908d632 | n/a          | 5.74                         | n/a       | 6.04         |                             |
| 286f830 | 5.12         | 5.33                         | 5.70      | 6.04         | Macro code generation       |
| 230bc34 | 5.22         | 5.80                         | 5.97      | 6.04         | Inline hints                |
| 71c68f5 | 5.40 (38.76) |                              | 6.28 (45) | 6.02 (43.15) | Ray far update optimization |

SanMiguel, max bounces 1, release-lto, average of two runs

`cargo run --profile release-lto -- -b 1 -m <mode>`

| Commit  | Packet4 | Packet4x2 (formerly Packet4) | Packet8      | Stream       |
| ------- | ------- | ---------------------------- | ------------ | ------------ |
| 908d632 | n/a     | 6.05                         | n/a          | 6.14         |
| 286f830 | 5.48    | 6.01                         | 6.18         | 6.13         |
| 230bc34 |         |                              | 6.16         | 6.13         |
| 71c68f5 |         |                              | 6.25 (44.86) | 6.15 (44.11) |
