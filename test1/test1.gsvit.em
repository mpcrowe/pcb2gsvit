/* Made with PCB Nelma export HID *//* Sat Jul  1 18:45:19 2017
 */
/* **** Nets **** */

net GND {
	objects = {
		"R1-2-bottom",
		"R1-2-top",
		"R1-2-outline"
	}
}
net mstrip1 {
	objects = {
		"R1-1-bottom",
		"R1-1-top",
		"R1-1-outline",
		"START-1-bottom",
		"START-1-top",
		"START-1-outline"
	}
}

/* **** Objects **** */

object R1-2-bottom {
	position = { 0, 0 }
	material = "copper"
	type = "image"
	role = "net"
	file = "test1.nelma.bottom.png"
	file-pos = { 132, 55 }
}
object R1-2-top {
	position = { 0, 0 }
	material = "copper"
	type = "image"
	role = "net"
	file = "test1.nelma.top.png"
	file-pos = { 132, 55 }
}
object R1-2-outline {
	position = { 0, 0 }
	material = "copper"
	type = "image"
	role = "net"
	file = "test1.nelma.outline.png"
	file-pos = { 132, 55 }
}
object R1-1-bottom {
	position = { 0, 0 }
	material = "copper"
	type = "image"
	role = "net"
	file = "test1.nelma.bottom.png"
	file-pos = { 132, 41 }
}
object R1-1-top {
	position = { 0, 0 }
	material = "copper"
	type = "image"
	role = "net"
	file = "test1.nelma.top.png"
	file-pos = { 132, 41 }
}
object R1-1-outline {
	position = { 0, 0 }
	material = "copper"
	type = "image"
	role = "net"
	file = "test1.nelma.outline.png"
	file-pos = { 132, 41 }
}
object START-1-bottom {
	position = { 0, 0 }
	material = "copper"
	type = "image"
	role = "net"
	file = "test1.nelma.bottom.png"
	file-pos = { 34, 42 }
}
object START-1-top {
	position = { 0, 0 }
	material = "copper"
	type = "image"
	role = "net"
	file = "test1.nelma.top.png"
	file-pos = { 34, 42 }
}
object START-1-outline {
	position = { 0, 0 }
	material = "copper"
	type = "image"
	role = "net"
	file = "test1.nelma.outline.png"
	file-pos = { 34, 42 }
}

/* **** Layers **** */

layer air-top {
	height = 90
	z-order = 1
	material = "air"
}
layer air-bottom {
	height = 90
	z-order = 1000
	material = "air"
}
layer bottom {
	height = 1
	z-order = 10
	material = "air"
	objects = {
		"R1-2-bottom",
		"R1-1-bottom",
		"START-1-bottom"
	}
}
layer substrate-11 {
	height = 45
	z-order = 11
	material = "composite"
}
layer top {
	height = 1
	z-order = 12
	material = "air"
	objects = {
		"R1-2-top",
		"R1-1-top",
		"START-1-top"
	}
}
layer substrate-13 {
	height = 45
	z-order = 13
	material = "composite"
}
layer outline {
	height = 1
	z-order = 14
	material = "air"
	objects = {
		"R1-2-outline",
		"R1-1-outline",
		"START-1-outline"
	}
}

/* **** Materials **** */

material copper {
	type = "metal"
	permittivity = 8.850000e-12
	conductivity = 0.0
	permeability = 0.0
}
material air {
	type = "dielectric"
	permittivity = 8.850000e-12
	conductivity = 0.0
	permeability = 0.0
}
material composite {
	type = "dielectric"
	permittivity = 3.894000e-11
	conductivity = 0.0
	permeability = 0.0
}

/* **** Space **** */

space pcb {
	step = { 1.270000e-04, 1.270000e-04, 3.500000e-05 }
	layers = {
		"air-top",
		"air-bottom",
		"bottom",
		"substrate-11",
		"top",
		"substrate-13",
		"outline"
	}
}
