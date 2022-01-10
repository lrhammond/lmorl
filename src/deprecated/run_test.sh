for ENV in BalanceBotEnv MountainCarContinuousSafe
do
for AGENT in LA2C LPPO
do
python3 ./src/main.py $AGENT $ENV 10000 2 &
done
done
