#!/usr/bin/env python3

class Data:
    product = {
        "milk": {"price": 1.50, "quantity": 10},
        "eggs": {"price": 0.20, "quantity": 100},
        "cheese": {"price": 2.00, "quantity": 10},
    }

    def __get__(self, obj, klas):
        return {"product": self.product}


class BusinessLogic:
    data = Data()

    def product_list(self):
        return self.data['product'].keys()

    def product_information(self, product):
        return self.data['product'].get(product, None)


class Ui:
    def __init__(self):
        self.business_logic = BusinessLogic()

    def get_product_list(self) -> None:
        print("PRODUCT LIST:")
        for product in self.business_logic.product_list():
            print(product)
        print()

    def get_product_information(self, product: str) -> None:
        product_info = self.business_logic.product_information(product)

        if product_info:
            print("PRODUCT INFORMATION:")
            print(f"Name: {product.title()}, " +
                  f"Price: {product_info.get('price', 0):.2f}, " +
                  f"Quantity: {product_info.get('quantity', 0)}")
        else:
            print(f"That product '{product}' does not exist in the records")
        print()


def main():
    """
    Separates presentation, application processing,
    and data management functions.
    """

    ui = Ui()

    ui.get_product_list()

    for product in ["cheese", "eggs", "milk", "arepas"]:
        ui.get_product_information(product)


if __name__ == "__main__":
    main()
