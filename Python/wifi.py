import subprocess

result = subprocess.run(
    ["netsh", "wlan", "show", "networks", "mode=bssid"],
    capture_output=True,
    text=True
)

print(result.stdout)