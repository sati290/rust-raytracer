SanMiguel, max bounces 1, release, average of two runs

`cargo run --release -- -b 1 -m <mode>`

| Commit  | Packet4 | Packet4x2 (formerly Packet4) | Packet8 | Stream |
| ------- | ------- | ---------------------------- | ------- | ------ |
| 908d632 | -       | 5.74                         | -       | 6.04   |
| 286f830 | 5.12    | 5.33                         | 5.70    | 6.04   |

SanMiguel, max bounces 1, release-lto, average of two runs

`cargo run --profile release-lto -- -b 1 -m <mode>`

| Commit  | Packet4 | Packet4x2 (formerly Packet4) | Packet8 | Stream |
| ------- | ------- | ---------------------------- | ------- | ------ |
| 908d632 | -       | 6.05                         | -       | 6.14   |
| 286f830 | 5.48    | 6.01                         | 6.18    | 6.13   |
