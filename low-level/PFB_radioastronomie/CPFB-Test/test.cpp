#include "pch.h"
#include "gtest/gtest.h"

TEST(Basic, FIR) {
	CriticalPolyphaseFilterbankTester CPF;
	EXPECT_EQ(1, 1);
	EXPECT_TRUE(true);
	bool some = CPF.FIR_c_reference();
	EXPECT_TRUE(some);
}
