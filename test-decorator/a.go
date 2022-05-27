package main

import "fmt"

func main() {
	var line string

	names := []string{"Sammy", "Jessica", "Drew", "Jamie"}

	line = join(",", names...)
	fmt.Println(line)
}

func join(del string, values ...string) string {
	var line string
	for i, v := range values {
		line = line + v
		if i != len(values)-1 {
			line = line + del
		}
	}
	return line
}

func main2() {
	sayHello()
	sayHello("Sammy")
	sayHello("Sammy", "Jessica", "Drew", "Jamie")
}

func sayHello(names ...string) {
	fmt.Println("say hello-----------")
	for _, n := range names {
		fmt.Printf("Hello %s\n", n)
	}
}
