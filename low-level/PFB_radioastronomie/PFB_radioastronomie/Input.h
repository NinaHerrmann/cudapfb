#pragma once
class Input
{
public:
	typedef	float * InputType;
	InputType Pointer;
public:
	Input();
	~Input();
	void setPointer(InputType Pointer);
	InputType getPointer();
};

