# 황새 게임 DQN / PPO 학습 프로젝트

실제 브라우저 게임에서 검출한 목 각도와 각도 변화량을 관측값으로 사용합니다.
행동은 `0=왼쪽 방향키`, `1=오른쪽 방향키`, `2=아무것도 하지 않음`입니다.

모델 초기화 시간 동안 먼저 시작된 게임은 학습에 포함하지 않습니다. 최초 게임은
방향키 입력 없이 종료 각도(현재 70도) 초과까지 추적한 후, 설정된 대기 시간과 재시작 절차를 거쳐
다음 게임을 학습 에피소드 1로 사용합니다.

## 파일 구성

```text
alganbird/
├─ stork_env.py          # Playwright + OpenCV 게임 환경
├─ training_config.py    # 보상, 에피소드, 스텝, 알고리즘 설정
├─ reward_wrapper.py     # 설정 가능한 보상 함수
├─ env_factory.py        # 환경 + 보상 + Monitor 조립
├─ training_common.py    # 체크포인트/에피소드 제한/설정 저장
├─ train_dqn.py          # DQN 학습
├─ train_ppo.py          # PPO 학습
├─ evaluate.py           # 저장 모델 평가
├─ models/               # 모델과 체크포인트
└─ logs/                 # Monitor CSV와 TensorBoard 로그
```

## 설치

PowerShell에서 프로젝트 폴더로 이동합니다.

```powershell
cd "C:\Users\07lee\hai_sub_project\alganbird"
python -m pip install -r requirements.txt
python -m playwright install chromium
```

## 설정 변경

`training_config.py`만 수정하면 됩니다.

- `ENV_CONFIG`: 5도 직립 조건, 60도 종료 조건, 키 입력 시간 등
- `REWARD_CONFIG`: 생존 보상, 직립 보너스, 각도 감점, 방향 행동 보상, 종료 감점
- `DQN_RUN_CONFIG`: DQN 총 스텝, 최대 에피소드, 저장 간격
- `DQN_CONFIG`: DQN 학습률, replay buffer, 탐험률 등
- `PPO_RUN_CONFIG`: PPO 총 스텝, 최대 에피소드, 저장 간격
- `PPO_CONFIG`: PPO rollout 길이, batch, epoch 등

### 직립 대기 중 각도 인식 복구

직립 대기 상태에서 각도 검출이 연속으로 실패하면 환경이 스페이스바를 한 번 눌러
화면 상태 복구를 시도합니다. 기본 설정은 3초 연속 미검출 시 1회 시도입니다.

```python
ENV_CONFIG = EnvironmentConfig(
    upright_detection_retry_seconds=3.0,
    upright_detection_max_retries=1,
    upright_fallen_retry_seconds=3.0,
    upright_fallen_max_retries=1,
)
```

- `upright_detection_retry_seconds`: 몇 초 동안 연속으로 각도를 찾지 못했을 때 복구할지 설정
- `upright_detection_max_retries`: 한 번의 직립 대기 과정에서 스페이스 복구를 시도할 최대 횟수
- `upright_detection_max_retries=0`: 자동 복구 비활성화
- `upright_fallen_retry_seconds`: 직립 대기 중 종료 각도(현재 70도)를 넘은 자세가 몇 초간 지속되면 복구할지 설정
- `upright_fallen_max_retries`: 넘어진 자세에 대한 스페이스 복구 최대 횟수
- `upright_fallen_max_retries=0`: 넘어진 자세 자동 복구 비활성화

각도가 정상 검출된 `5~70도` 구간에서는 아직 직립 상태로 돌아오는 중인 것으로 보고
복구용 스페이스를 누르지 않습니다. 하지만 70도를 넘은 자세가 설정 시간 동안
유지되면 게임이 넘어진 화면에 고정된 것으로 판단하고 복구 스페이스를 누릅니다.
복구가 실행되면 콘솔에 `[직립 복구]` 상태가 출력됩니다. 기존 게임
준비·시작·종료 후 재시작에 사용되는 스페이스 입력 순서는 그대로 유지됩니다.

### 방향 행동 보상

목 각도를 0도 방향으로 되돌리는 키 입력에 추가 보상을 줍니다. 현재 설정값은
올바른 행동 `+2.0`, 잘못된 행동 `-1.0`입니다. 중심 근처에서는 방향키를
계속 누르지 않도록 무행동을 올바른 행동으로 처리합니다.

| 현재 목 각도 | 올바른 행동 | 행동 번호 | 방향 보상 |
|---|---|---:|---:|
| `+2도` 초과 | 왼쪽 방향키 | `0` | `+2.0` |
| `-2도` 미만 | 오른쪽 방향키 | `1` | `+2.0` |
| `±2도` 이내 | 아무것도 하지 않음 | `2` | `+2.0` |

위 상태에서 다른 행동을 선택하면 `-1.0`이 적용됩니다. 따라서 무행동을 2도 밖에서
선택하는 경우에도 잘못된 행동으로 감점됩니다. 아래 설정으로 보상 강도와
0도 주변의 무시 구간을 조절할 수 있습니다.

```python
REWARD_CONFIG = RewardConfig(
    correct_direction_reward=2.0,
    wrong_direction_penalty=-1.0,
    direction_dead_zone=2.0,
)
```

`total_timesteps`와 `max_episodes` 중 먼저 도달한 조건에서 학습이 끝납니다.
PPO는 rollout 단위로 데이터를 모으기 때문에 실제 종료 스텝이 설정값보다 조금 클 수 있습니다.

처음 테스트할 때는 아래처럼 작게 시작하는 것이 좋습니다.

```python
DQN_RUN_CONFIG = RunConfig(
    total_timesteps=5_000,
    max_episodes=50,
)
```

긴 학습에서는 콘솔 출력 부담을 줄이도록 다음 값을 권장합니다.

```python
ENV_CONFIG = EnvironmentConfig(print_status=False)
```

`progress_bar`는 Rich/tqdm이 Python 종료 시 부가 오류를 내는 환경을 피하기 위해
기본값이 `False`입니다. 학습 상태는 SB3 기본 로그와 TensorBoard에서 확인할 수 있습니다.

## 기존 모델 호환성

행동 공간이 2개에서 3개로 변경되었으므로, 이전의 2행동 모델은 현재 환경과 호환되지
않습니다. DQN과 PPO 모두 무행동을 학습하려면 새 모델로 학습을 시작해야 합니다.

## DQN 학습

```powershell
python train_dqn.py
```

## PPO 학습

```powershell
python train_ppo.py
```

DQN과 PPO를 동시에 실행하면 두 브라우저가 키 입력과 자원을 경쟁할 수 있으므로 하나씩
실행하는 것을 권장합니다.

## 저장 결과

실행할 때마다 날짜와 시각으로 새 폴더가 생성됩니다.

```text
models/dqn/20260724_120000/final_model.zip
models/ppo/20260724_130000/final_model.zip
```

각 모델 폴더에는 학습 당시의 `config_snapshot.json`도 저장됩니다. 체크포인트는
`checkpoints/`에 저장되고, DQN은 replay buffer도 함께 저장합니다.

## TensorBoard

```powershell
tensorboard --logdir logs
```

표시된 주소를 브라우저에서 열면 보상과 손실 변화를 확인할 수 있습니다.

## 모델 평가

```powershell
python evaluate.py --algorithm dqn --model "models\dqn\실행폴더\final_model.zip" --episodes 10
```

```powershell
python evaluate.py --algorithm ppo --model "models\ppo\실행폴더\final_model.zip" --episodes 10
```

## 이어서 학습

`training_config.py`의 해당 실행 설정에 모델 경로를 넣습니다.

```python
DQN_RUN_CONFIG = RunConfig(
    total_timesteps=100_000,
    max_episodes=1_000,
    resume_model_path=r"models\dqn\실행폴더\final_model.zip",
)
```

그 후 동일하게 `python train_dqn.py`를 실행합니다.
