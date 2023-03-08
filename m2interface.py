import subprocess


def solve(panels):
    argms = [
        "M2",
        "--script",
        "modules.m2",
        createIdealString(panels[0]),
        createIdealString(panels[1]),
        createIdealString(panels[2]),
        createIdealString(panels[3]),
        createIdealString(panels[4]),
        createIdealString(panels[5]),
        createIdealString(panels[6]),
        createIdealString(panels[7]),
        createIdealString(panels[8]),
        createIdealString(panels[9]),
        createIdealString(panels[10]),
        createIdealString(panels[11]),
        createIdealString(panels[12]),
        createIdealString(panels[13]),
        createIdealString(panels[14]),
        createIdealString(panels[15])
    ]
    try:
        result = (
            subprocess.check_output(argms).decode("utf-8").replace('\n', '').replace('}', '').replace('{', '').replace(
                ' ', '').split(','))
        return result
    except subprocess.CalledProcessError as e:
        print(e.returncode)
        print(e.output)


def createIdealString(panel):
    idealStr = ""
    for comp in panel:
        idealStr = idealStr + "," + str(comp).replace(str(comp)[0], 'x_').replace(', ', '*x_').replace(']', '')
    return idealStr[1:]
