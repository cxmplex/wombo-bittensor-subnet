PORT=33022  CUDA_VISIBLE_DEVICES=0 REDIS_URI="redis://198.166.137.3:20004" pm2 start poetry --name wombo-generator0 --interpreter none -- run python miner/main.py --netuid 30 --wallet.name ben_wallet --wallet_hotkey hk00
sleep 20;
PORT=33067  CUDA_VISIBLE_DEVICES=1 REDIS_URI="redis://198.166.137.3:20004" pm2 start poetry --name wombo-generator1 --interpreter none -- run python miner/main.py --netuid 30 --wallet.name ben_wallet --wallet_hotkey hk00
sleep 20;
PORT=33076  CUDA_VISIBLE_DEVICES=2 REDIS_URI="redis://198.166.137.3:20004" pm2 start poetry --name wombo-generator2 --interpreter none -- run python miner/main.py --netuid 30 --wallet.name ben_wallet --wallet_hotkey hk00
sleep 20;
PORT=33097  CUDA_VISIBLE_DEVICES=3 REDIS_URI="redis://198.166.137.3:20004" pm2 start poetry --name wombo-generator3 --interpreter none -- run python miner/main.py --netuid 30 --wallet.name ben_wallet --wallet_hotkey hk00
sleep 20;
PORT=33135  CUDA_VISIBLE_DEVICES=4 REDIS_URI="redis://198.166.137.3:20004" pm2 start poetry --name wombo-generator4 --interpreter none -- run python miner/main.py --netuid 30 --wallet.name ben_wallet --wallet_hotkey hk00
sleep 20;
PORT=33146  CUDA_VISIBLE_DEVICES=5 REDIS_URI="redis://198.166.137.3:20004" pm2 start poetry --name wombo-generator5 --interpreter none -- run python miner/main.py --netuid 30 --wallet.name ben_wallet --wallet_hotkey hk00
sleep 20;
PORT=33166  CUDA_VISIBLE_DEVICES=6 REDIS_URI="redis://198.166.137.3:20004" pm2 start poetry --name wombo-generator6 --interpreter none -- run python miner/main.py --netuid 30 --wallet.name ben_wallet --wallet_hotkey hk00
sleep 20
PORT=33192  CUDA_VISIBLE_DEVICES=7 REDIS_URI="redis://198.166.137.3:20004" pm2 start poetry --name wombo-generator7 --interpreter none -- run python miner/main.py --netuid 30 --wallet.name ben_wallet --wallet_hotkey hk00
