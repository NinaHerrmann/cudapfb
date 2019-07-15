cd C:\PFB\low-level\PFB_radioastronomie\x64\Debug
PFB_radioastronomie.exe 
:: 2. Read timestamp from timestamp.txt
cd C:\Users\b98\PFB_OUTPUT
:: @ping -n 10 localhost> nul
if exist timestamp.txt set /p timestamp=<timestamp.txt
cd %timestamp%
:: 2. Read settings from setting file and start python script from Ubuntu App with $timestamp $taps $channels $spectra
for /f "tokens=1,2,3,4 delims==;" %%G in (settings) do (
	set %%G=%%H
	wsl python /root/pfb.py %timestamp% %%I %%H %%J
)
:: Compare output
wsl python /root/filecomparison.py %timestamp%
cd C:\