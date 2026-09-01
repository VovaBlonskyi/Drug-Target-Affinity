const proc = Bun.spawn([
    "python",
    "predict.py",
    "N[C@H](C(=O)O)CC1=CNC2=CC=CC=C12",
    "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"
])

const output = Number(await new Response(proc.stdout).text())

console.log(output )