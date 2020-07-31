const express = require("express")
const { spawn } = require("child_process");

const app = express()
const PORT = process.env.PORT || 8000;

app.use(express.urlencoded());
app.use(express.json());

app.use("/",express.static("public/finloans"));

app.post("/submit",(req,res) => {
	res.send(req.body.name);
});

app.get("/python",(req,res) => {
	const process = spawn("C:/ProgramData/Anaconda3/python",["scripts/script.py"]);
	process.stdout.on("data", data => res.json({result:data.toString()}));
});

app.get("/empreinte",(req,res) => {
	const process = spawn("C:/ProgramData/Anaconda3/python",["scripts/fingerprint_detection.py"]);
	process.stdout.on("data", data => res.json({result:data.toString()}));
});


app.listen(PORT,() => console.log(`server listening on port ${PORT}`));
