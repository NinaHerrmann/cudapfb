#include "Input.h"

Input::Input()
{
}


Input::~Input()
{
}
void Input::setPointer(InputType newPointer) {
	Pointer = newPointer;
}
float * Input::getPointer() {
	return Pointer;
}